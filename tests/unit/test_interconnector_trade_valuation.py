"""Item IC1: every post-solve valuation of cross-border trade uses the LP's own curve.

Before IC1 the FTI settlement and the nationalised institutions' link measurement
valued a signed interconnector's flow at its import ask in both directions, so an
export earned 89.93 GBP/MWh where the LP paid it about 38.6 (efficiency times the
CP.29 export bid). That was most of the central dispatch "resource saving" and
the institution effect on the nationalised bill. Every consumer now calls
``market_coupling.interconnector_trade_cost``, which the tranche tests show
reproduces the LP objective.
"""

import logging

import pandas as pd
import pypsa
import pytest

from scripts.interconnectors.add_to_network import add_interconnector_links
from scripts.interconnectors.market_coupling import build_piecewise_interconnector_cost_callback
from scripts.market.analyze_market import calculate_interconnector_metrics
from scripts.market.fti_settlement import link_trade_cost

IDX = pd.date_range("2030-01-01", periods=3, freq="h")


def _links(**tranche):
    links = pd.DataFrame({
        "marginal_cost": [90.0, 5.0],
        "efficiency": [0.975, 1.0],
        "foreign_market_pricing": ["piecewise_bidirectional", ""],
        "interconnector_name": ["ic", ""],
        "market_separate_import_export_prices": [True, False],
        "market_export_bid_offset": [50.0, 0.0],
        "market_spread_half": [0.5, 0.0],
        "market_circulation_epsilon": [0.01, 0.0],
    }, index=["IC_ic", "internal"])
    for column, value in tranche.items():
        links[column] = [value, None]
    return links


def test_exports_earn_the_export_bid_not_the_import_ask():
    p0 = pd.DataFrame({"IC_ic": [100.0, -50.0, 0.0], "internal": [10.0] * 3, "unpriced": [999.0] * 3}, index=IDX)
    # import 100 at the 90 ask; export 50 earns 0.975^2 x 40 = 38.025 (loss paid, item D2); internal 30 at 5
    assert link_trade_cost(p0, _links()) == pytest.approx(100 * 90 - 50 * 0.975 ** 2 * 40 + 30 * 5)


def test_the_tranche_cost_is_part_of_the_valuation():
    links = _links(market_tranche_group="continent", market_tranche_width_mw=40.0, market_tranche_count=3,
                   market_tranche_import_step=2.0, market_tranche_export_step=1.0)
    p0 = pd.DataFrame({"IC_ic": [100.0, -50.0, 0.0], "internal": [0.0] * 3}, index=IDX)
    base = 100 * 90 - 50 * 0.975 ** 2 * 40
    tranches = 2.0 * (100 - 40) + 2.0 * (100 - 80) + 1.0 * (50 - 40)
    assert link_trade_cost(p0, links) == pytest.approx(base + tranches)


def test_hourly_asks_are_read_when_the_network_carries_them():
    ask_t = pd.DataFrame({"IC_ic": [100.0, 60.0, 80.0]}, index=IDX)
    p0 = pd.DataFrame({"IC_ic": [10.0, -10.0, 0.0]}, index=IDX)
    assert link_trade_cost(p0, _links(), ask_t) == pytest.approx(10 * 100 - 10 * 0.975 ** 2 * (60 - 50))


def _solved_network(tmp_path, tranche_step=0.0):
    network = pypsa.Network()
    network.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    network.add("Bus", "GB")
    network.add("Bus", "External_France")
    cables = pd.DataFrame([{"name": "C", "from_bus": "GB", "to_bus": "External_France", "capacity_mw": 100.0,
                            "losses_percent": 10.0, "counterparty_country": "France"}])
    prices = pd.DataFrame([{"country": "france", "year": 2030, "pathway": "Central",
                            "estimated_price_gbp_per_mwh": 20.0}])
    add_interconnector_links(network, cables, pd.DataFrame(), prices, modelled_year=2030, fes_pathway="Central",
                             pricing_config={"transaction_spread_gbp_per_mwh": 1.0})
    network.links.loc["IC_C", "market_tranche_group"] = "continent"
    network.links.loc["IC_C", "market_tranche_width_mw"] = 30.0
    network.links.loc["IC_C", "market_tranche_count"] = 3
    network.links.loc["IC_C", "market_tranche_import_step"] = tranche_step
    network.links.loc["IC_C", "market_tranche_export_step"] = tranche_step
    network.add("Load", "demand", bus="GB", p_set=[45.0, 0.0])
    network.add("Generator", "domestic", bus="GB", p_nom=500.0, marginal_cost=[1000.0, 0.0])
    status, _ = network.optimize(solver_name="highs",
                                 extra_functionality=build_piecewise_interconnector_cost_callback(network))
    assert status == "ok"
    return network


def test_measure_link_trade_values_the_saved_network_on_the_lp_curve(tmp_path, monkeypatch):
    from scripts.public_ownership import nationalised_institutions as ni

    network = _solved_network(tmp_path, tranche_step=3.0)
    monkeypatch.setattr(ni, "MARKET_DIR", tmp_path)
    network.export_to_netcdf(tmp_path / "S_balancing.nc")
    measured = ni.measure_link_trade(["S"])
    generator_cost = float((network.generators_t.p * network.generators_t.marginal_cost).sum().sum())
    assert measured["interconnector_cost_gbp"] == pytest.approx(network.objective - generator_cost, rel=1e-6)
    # Hour 1 imports, hour 2 exports: a flat ask on both would net them off.
    flows = network.links_t.p0["IC_C"]
    assert flows.iloc[0] > 0 > flows.iloc[1]


def test_retrading_and_rent_include_the_tranche_cost(tmp_path):
    from scripts.interconnectors.market_coupling import interconnector_trade_cost, link_price_tables
    from scripts.market.analyze_market import calculate_interconnector_retrading_metrics

    network = _solved_network(tmp_path, tranche_step=3.0)
    snapshots = network.snapshots
    day_ahead = pd.DataFrame({"IC_C": [0.0, 0.0]}, index=snapshots)
    balancing = network.links_t.p0[["IC_C"]]
    links, ask_t = link_price_tables(network)
    valued = interconnector_trade_cost(balancing, links, ask_t)["IC_C"]

    retrading = calculate_interconnector_retrading_metrics(network, day_ahead, balancing)
    assert float(retrading["retrading_cost_GBP"].sum()) == pytest.approx(float(valued.sum()), rel=1e-9)

    gb = pd.Series([50.0, 50.0], index=snapshots)
    rent = calculate_interconnector_metrics(network, balancing, gb, logging.getLogger(__name__))
    flow = balancing["IC_C"]
    delivered_import = flow.clip(lower=0.0) * 0.9
    # GB gives up efficiency x |p0| on an export: PyPSA books p1 = -efficiency x p0 (item B9).
    sent_export = (-flow).clip(lower=0.0) * 0.9
    expected_rent = float((gb * (delivered_import - sent_export) - valued).sum())
    assert float(rent["estimated_congestion_rent_GBP"].sum()) == pytest.approx(expected_rent, rel=1e-9)


def test_the_diagnostic_reports_realised_prices_and_tranche_occupancy():
    from scripts.market.diagnose_interconnector_pricing import trade_price_summary

    links = _links(market_tranche_group="continent", market_tranche_width_mw=40.0, market_tranche_count=3,
                   market_tranche_import_step=2.0, market_tranche_export_step=1.0)
    links["p_nom"] = [120.0, 10.0]
    links["market_import_level"] = [89.43, None]
    links["market_export_level"] = [40.09, None]
    idx = pd.date_range("2030-01-01", periods=4, freq="h")
    p0 = pd.DataFrame({"IC_ic": [30.0, 100.0, -50.0, -20.0], "internal": [5.0] * 4}, index=idx)
    row = trade_price_summary(p0, links).loc["continent"]
    assert row["import_mwh"] == pytest.approx(130.0)
    assert row["export_mwh"] == pytest.approx(70.0)
    # 100 MW in hour 2 is in the third tranche: 90 + two steps on the part above each boundary.
    assert row["import_price_paid"] == pytest.approx((130 * 90 + 2 * 60 + 2 * 20) / 130)
    assert row["import_hours_tranche_1"] == 1 and row["import_hours_tranche_3"] == 1
    assert row["export_hours_tranche_2"] == 1 and row["export_hours_tranche_1"] == 1
    # 50 MW exported loses one step on the 10 MW above the first boundary.
    assert row["export_price_received"] == pytest.approx((70 * 0.975 ** 2 * 40 - 1.0 * 10) / 70)
    assert row["export_price_hourly_std"] > 0.0


def test_tables_read_from_a_saved_network_carry_its_timestamps(tmp_path):
    """PyPSA's NetCDF indexes snapshots 0..n and keeps the times in snapshots_snapshot.

    An integer index silently fails to join the Stage 1 flow CSV (timestamped), and
    every hour then falls back to the static ask: caught on the IC1 gate week.
    """
    import xarray as xr

    from scripts.interconnectors.market_coupling import link_price_tables_from_dataset

    network = _solved_network(tmp_path, tranche_step=3.0)
    network.links_t.marginal_cost = pd.DataFrame({"IC_C": [30.0, 10.0]}, index=network.snapshots)
    path = tmp_path / "net.nc"
    network.export_to_netcdf(path)
    with xr.open_dataset(path) as ds:
        links, ask_t = link_price_tables_from_dataset(ds)
    assert ask_t.index.equals(pd.DatetimeIndex(network.snapshots))
    assert ask_t["IC_C"].tolist() == [30.0, 10.0]
