"""The bill projector counts demand carried by flexibility units as GB electrical load (2 Oct 2026).

The 2050 build represents consumer flexibility (heat pumps, EVs, district heat, V2G,
DACCS) as storage units whose negative inflow is the demand they serve. That demand is
metered supply, so it joins the recovery denominators. A network without negative
inflow (every 2030 build) is unchanged.
"""

import numpy as np
import pandas as pd
import pypsa

from scripts.consumer_bill.project_future_bill import read_hourly_inputs


def _network(tmp_path, flex_inflow=None, hydro_inflow=None):
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2050-01-01", periods=4, freq="h"))
    n.add("Bus", "GB1", carrier="AC")
    n.add("Load", "L1", bus="GB1", p_set=[100.0, 120.0, 90.0, 110.0])
    if flex_inflow is not None:
        n.add("StorageUnit", "GB1 HP_flexibility", bus="GB1", carrier="HP_flexibility", p_nom=50.0,
              max_hours=4.0, inflow=flex_inflow)
    if hydro_inflow is not None:
        n.add("StorageUnit", "GB1 hydro", bus="GB1", carrier="large_hydro", p_nom=50.0,
              max_hours=10.0, inflow=hydro_inflow)
    path = tmp_path / "net.nc"
    n.export_to_netcdf(path)
    prices = pd.DataFrame({"wholesale_price": [50.0, 60.0, 40.0, 55.0]}, index=n.snapshots)
    prices.index.name = "snapshot"
    price_path = tmp_path / "price.csv"
    prices.to_csv(price_path)
    return path, price_path


def test_negative_inflow_joins_the_gb_electrical_load(tmp_path):
    flex = [-10.0, -20.0, -5.0, 0.0]
    net, price = _network(tmp_path, flex_inflow=flex, hydro_inflow=[30.0, 30.0, 30.0, 30.0])
    h = read_hourly_inputs(net, price)
    expected = np.array([110.0, 140.0, 95.0, 110.0])
    np.testing.assert_allclose(h["gb_electrical_load_mw"].to_numpy(), expected)
    np.testing.assert_allclose(h["total_load_mw"].to_numpy(), expected)
    np.testing.assert_allclose(h["demand_mw"].to_numpy(), expected)
    # Positive inflow (hydro) is not demand; the hydrogen offtake stays the non-electrical rest.
    np.testing.assert_allclose(h["h2_offtake_mw"].to_numpy(), 0.0)
    # The market summaries count Loads only; their checks subtract this (no extra column).
    assert h.attrs["flexible_demand_mwh"] == 35.0
    assert "flexible_demand_mw" not in h.columns


def test_network_without_negative_inflow_is_unchanged(tmp_path):
    plain_dir, hydro_dir = tmp_path / "plain", tmp_path / "hydro"
    plain_dir.mkdir()
    hydro_dir.mkdir()
    plain_net, plain_price = _network(plain_dir)
    hydro_net, hydro_price = _network(hydro_dir, hydro_inflow=[30.0, 30.0, 30.0, 30.0])
    pd.testing.assert_frame_equal(read_hourly_inputs(plain_net, plain_price),
                                  read_hourly_inputs(hydro_net, hydro_price))
