"""Consumer flexibility is demand, not storage plant (F5 follow-up, 27 Oct 2026).

The flexibility storage units (EVs, V2G, heat, storage heaters, DSR, DACCS) carve their energy out
of the Loads, so every demand total, demand-weighted price and settlement counts what they draw
(p_store - p_dispatch) as demand and never treats them as an asset or a storage fleet.
"""
import logging

import numpy as np
from pathlib import Path

import pandas as pd
import pypsa
import pytest

from scripts.demand import demand_carriers as dc
from scripts.market import market_accounting as ma

LOG = logging.getLogger("test_consumer_flex_is_demand")
IDX = pd.date_range("2030-01-01", periods=4, freq="h")


def _network(solved=True):
    n = pypsa.Network()
    n.set_snapshots(IDX)
    n.add("Bus", "b", carrier="AC", country="GB")
    n.add("Load", "load_b", bus="b", carrier="electricity", p_set=pd.Series([100.0, 100.0, 100.0, 100.0], index=IDX))
    n.add("StorageUnit", "bat", bus="b", carrier="Battery", p_nom=50.0, max_hours=2.0,
          efficiency_store=0.9, efficiency_dispatch=0.9)
    n.add("StorageUnit", "b EV_smart_charging", bus="b", carrier=dc.EV_SMART_CHARGING, p_nom=40.0, max_hours=1.0,
          efficiency_store=0.9, inflow=pd.Series([-9.0, -9.0, 0.0, 0.0], index=IDX))
    n.add("StorageUnit", "b IC_DSR", bus="b", carrier=dc.IC_DSR, p_nom=10.0, max_hours=2.0)
    if solved:
        n.storage_units_t.p_store = pd.DataFrame({"bat": [5.0, 5.0, 0.0, 0.0], "b EV_smart_charging": [20.0, 0.0, 0.0, 0.0],
                                                  "b IC_DSR": [0.0, 4.0, 0.0, 0.0]}, index=IDX)
        n.storage_units_t.p_dispatch = pd.DataFrame({"bat": [5.0, 0.0, 4.0, 0.0], "b EV_smart_charging": 0.0,
                                                     "b IC_DSR": [0.0, 0.0, 4.0, 0.0]}, index=IDX)
        n.storage_units_t.p = n.storage_units_t.p_dispatch - n.storage_units_t.p_store
    return n


def test_flex_draw_reads_charging_net_of_discharge_and_ignores_plant():
    n = _network()
    draw = ma._network_flex_draw(n, n.snapshots)
    assert set(draw.columns) == {"b EV_smart_charging", "b IC_DSR"}
    assert draw.sum(axis=1).tolist() == [20.0, 4.0, -4.0, 0.0]
    # -p when only the net series is saved; a unit missing from a saved series is zero, never its drain
    su = n.storage_units
    p_only = ma.consumer_flex_draw(su["carrier"], p=n.storage_units_t.p)
    assert p_only.sum(axis=1).tolist() == [20.0, 4.0, -4.0, 0.0]
    partial = ma.consumer_flex_draw(su["carrier"], p_store=n.storage_units_t.p_store[["b EV_smart_charging"]])
    assert partial["b IC_DSR"].sum() == 0.0
    # a built network: the drain's electricity, over the charging efficiency
    built = _network(solved=False)
    draw = ma._network_flex_draw(built, built.snapshots)
    assert draw["b EV_smart_charging"].tolist() == pytest.approx([10.0, 10.0, 0.0, 0.0])


def test_accounting_demand_counts_flexibility_and_reports_it():
    n = _network()
    hours = pd.Series(1.0, index=IDX)
    assert ma._demand_energy_mwh(n, hours) == pytest.approx(400.0 + 20.0)
    assert ma._gb_electricity_demand_mwh(n, hours) == pytest.approx(420.0)
    accounting, _ = ma.create_market_accounting_report(n, pd.DataFrame())
    assert accounting["consumer_flexibility_MWh"] == pytest.approx(20.0)
    assert accounting["demand_MWh"] - accounting["consumer_flexibility_MWh"] == pytest.approx(400.0)


def test_storage_burn_leaves_consumer_flexibility_out():
    n = _network()
    n.storage_units_t.p_store.loc[IDX[2], "b IC_DSR"] = 4.0           # a DSR overlap: load shifting, not burn
    burn = ma.storage_simultaneous_burn(n, pd.Series(1.0, index=IDX), tol_pu=0.0)
    assert set(burn["burn_by_carrier_MWh"]) == {"Battery"}


def test_bill_reads_flexibility_when_it_buys(tmp_path):
    from scripts.consumer_bill.project_future_bill import _read_hourly_inputs

    n = _network()
    path = tmp_path / "w.nc"
    n.export_to_netcdf(path)
    prices = tmp_path / "p.csv"
    pd.DataFrame({"wholesale_price": [10.0, 20.0, 30.0, 40.0]}, index=pd.Index(IDX, name="snapshot")).to_csv(prices)
    out = _read_hourly_inputs(path, prices)
    assert out["demand_mw"].tolist() == pytest.approx([120.0, 104.0, 96.0, 100.0])
    assert out.attrs["flexible_demand_mwh"] == pytest.approx(20.0)
    assert (out["total_load_mw"] - out["gb_electrical_load_mw"]).abs().max() == pytest.approx(0.0)


def test_fti_demand_weights_include_flexibility():
    from scripts.market.fti_settlement import _gb_loads

    load, bus = _gb_loads(_network())
    assert {"load_b", "b EV_smart_charging", "b IC_DSR"} <= set(load.columns)
    assert load.sum(axis=1).tolist() == pytest.approx([120.0, 104.0, 96.0, 100.0])
    assert bus["b EV_smart_charging"] == "b"


def test_demand_weighted_price_includes_flexibility():
    from scripts.market.analyze_market import _demand_weighted_wholesale_price

    price = pd.Series([10.0, 20.0, 30.0, 40.0], index=IDX)
    got = _demand_weighted_wholesale_price(price, _network(), LOG)
    assert got == pytest.approx(float((price * np.array([120.0, 104.0, 96.0, 100.0])).sum() / 420.0))


def test_2030_register_leaves_consumer_flexibility_out_and_2050_drops_it_later():
    from scripts.public_ownership.build_asset_finance_register import _component_rows

    n = _network()
    rows_2030 = pd.DataFrame(_component_rows(n, "StorageUnit", 2030, 2028))
    assert set(rows_2030["carrier"]) == {"Battery"}
    rows_2050 = pd.DataFrame(_component_rows(n, "StorageUnit", 2050, 2028))
    assert dc.EV_SMART_CHARGING in set(rows_2050["carrier"])          # vintage_register excludes and reports it


def test_settlement_and_market_filters_share_one_carrier_list():
    from scripts.public_ownership import settlement

    # The notebook builder writes its notebook when imported, so read its text instead.
    builder = Path("scripts/analysis/build_cp30_bill_notebook.py").read_text(encoding="utf-8")
    assert "EXCLUDE |= set(DEMAND_FLEX_STORAGE_CARRIERS)" in builder

    assert settlement.DEMAND_FLEX_STORAGE_CARRIERS is dc.DEMAND_FLEX_STORAGE_CARRIERS
    assert dc.STORAGE_HEATER_FLEXIBILITY in dc.DEMAND_FLEX_STORAGE_CARRIERS and dc.DACCS in dc.DEMAND_FLEX_STORAGE_CARRIERS


def test_the_bm_holds_consumer_flexibility_at_stage_1_whatever_the_participation_lists_say():
    """F5: a family whose exclude_carriers omits the flex carriers still cannot redispatch them."""
    import logging

    import pandas as pd
    import pypsa

    from scripts.market.market_utils import _apply_participation_policy

    n = pypsa.Network()
    n.add("Bus", "b")
    n.add("Generator", "g", bus="b", carrier="CCGT", p_nom=10)
    n.add("StorageUnit", "battery", bus="b", carrier="battery", p_nom=5)
    n.add("StorageUnit", "ev", bus="b", carrier="EV_smart_charging", p_nom=5)
    n.add("StorageUnit", "dsr", bus="b", carrier="IC_DSR", p_nom=2)
    offer = pd.Series(50.0, index=n.generators.index)
    su = pd.Series(80.0, index=n.storage_units.index)
    for behavior in ("priced_out", "fixed"):
        cfg = {"balancing": {"participation": {"storage_units": {"behavior": behavior, "exclude_carriers": []}}}}
        _apply_participation_policy(n, cfg, logging.getLogger("t"), offer.copy(), offer.copy(), su.copy(), su.copy())
        assert list(n._bm_participating_storage_units) == ["battery"], behavior
        assert list(n._bm_fixed_storage_units) == ["ev", "dsr"], behavior
        assert list(n._bm_eligible_storage_units) == ["battery"], behavior
