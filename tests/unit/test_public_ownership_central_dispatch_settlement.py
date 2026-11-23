"""P1/P2 under central dispatch (20 Oct 2026): the institution twin's asset settlement on the I3 year.

Each asset is paid the central schedule at the national price (the market twin's copperplate
price), its own FTI transfers in the balancing settlement (cost recovery, firm access with the
lost support, the storage make-whole, its real-time correction at cost), CfD and ROC support on
metered central output; the lost top-up is reported, never added.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from scripts.public_ownership.settlement import (
    CENTRAL_TRANSFER_COLUMNS, SETTLEMENT_COLUMNS, _central_month_settlement, central_month_units,
    settle_assets_central_dispatch,
)

SNAPSHOTS = pd.date_range("2030-01-01", periods=3, freq="h")
GENERATORS = ["gas", "wind", "roc_wind", "EU_x"]
STORES = ["bat"]


def _network(path, gen_p, store_p):
    """A small saved network in PyPSA's NetCDF layout: four generators, one store, two buses."""
    n = len(SNAPSHOTS)
    ds = xr.Dataset(
        {
            "snapshots_snapshot": ("snapshots", SNAPSHOTS.values),
            "snapshots_generators": ("snapshots", np.ones(n)),
            "generators_carrier": ("generators_i", ["CCGT", "wind_offshore", "wind_onshore", "EU_market"]),
            "generators_support_type": ("generators_i", ["", "CfD", "ROC", ""]),
            "generators_cfd_round": ("generators_i", ["", "AR4", "", ""]),
            "generators_ro_banding": ("generators_i", [np.nan, np.nan, 0.9, np.nan]),
            "generators_bus": ("generators_i", ["A", "B", "B", "A"]),
            "generators_marginal_cost": ("generators_i", [80.0, 0.0, -50.0, 0.0]),
            "generators_t_p": (("snapshots", "generators_t_p_i"), np.asarray(gen_p, dtype=float)),
            "storage_units_carrier": ("storage_units_i", ["battery"]),
            "storage_units_bus": ("storage_units_i", ["A"]),
            "storage_units_t_p": (("snapshots", "storage_units_t_p_i"), np.asarray(store_p, dtype=float)),
            "buses_lat": ("buses_i", [52.0, 57.0]),
            "buses_lon": ("buses_i", [-1.0, -3.0]),
        },
        coords={"snapshots": np.arange(n), "generators_i": GENERATORS, "generators_t_p_i": GENERATORS,
                "storage_units_i": STORES, "storage_units_t_p_i": STORES, "buses_i": ["A", "B"]},
    )
    ds.to_netcdf(path)
    return path


def _months(tmp_path):
    sched = [[100.0, 50.0, 20.0, 0.0], [100.0, 0.0, 20.0, 10.0], [0.0, 80.0, 20.0, 0.0]]
    phys = [[110.0, 50.0, 20.0, 0.0], [100.0, 0.0, 10.0, 10.0], [0.0, 80.0, 20.0, 0.0]]
    w = _network(tmp_path / "C_Jan_wholesale.nc", sched, [[5.0], [-5.0], [0.0]])
    b = _network(tmp_path / "C_Jan_balancing.nc", phys, [[5.0], [-5.0], [0.0]])
    return w, b


def _units():
    return pd.DataFrame({
        "month": "Jan", "central_scenario": "C_Jan", "market_scenario": "M_Jan",
        "component_type": ["Generator", "Generator", "Generator", "StorageUnit"],
        "asset_id": ["gas", "wind", "EU_x", "bat"], "carrier": ["CCGT", "wind_offshore", "EU_market", "battery"],
        "cost_recovery_gbp": [1000.0, 0.0, 50.0, 0.0], "firm_access_margin_gbp": [0.0, 400.0, 0.0, 0.0],
        "firm_access_cfd_support_gbp": [0.0, 300.0, 0.0, 0.0], "firm_access_roc_support_gbp": [0.0, 0.0, 0.0, 0.0],
        "storage_make_whole_gbp": [0.0, 0.0, 0.0, 25.0]})


PRICE = pd.Series([60.0, 70.0, 10.0], index=SNAPSHOTS)
FUEL = pd.Series({"gas": 75.0, "wind": 0.0, "roc_wind": 0.0})


def test_each_asset_is_paid_its_own_transfers_and_the_central_schedule(tmp_path):
    w, b = _months(tmp_path)
    month = _central_month_settlement(w, b, PRICE, central_month_units(_units(), "C_Jan"), {"AR4": 90.0}, None,
                                      roc_value_gbp_per_roc=50.0, fuel_rates=FUEL).set_index("asset_id")
    gas, wind = month.loc["gas"], month.loc["wind"]
    # The central schedule at the national price.
    assert gas["wholesale_revenue_gbp"] == pytest.approx(100 * 60 + 100 * 70)
    # Real-time correction at its own fuel rate: 10 MWh more in hour one at 75.
    assert gas["realtime_at_cost_gbp"] == pytest.approx(750.0)
    assert gas["balancing_settlement_gbp"] == pytest.approx(1000.0 + 750.0)
    assert gas["fuel_carbon_gbp"] == pytest.approx((110 + 100) * 75.0)
    # Firm access with the lost top-up inside the balancing settlement; the CfD line on metered output.
    assert wind["balancing_settlement_gbp"] == pytest.approx(700.0)
    assert wind["cfd_curtailment_compensation_gbp"] == pytest.approx(300.0)
    assert wind["cfd_support_gbp"] == pytest.approx(50 * (90 - 60) + 80 * (90 - 10))
    # ROCs on metered central output (10 + 20 + 20 MWh at 0.9 x 50).
    assert month.loc["roc_wind", "roc_support_gbp"] == pytest.approx(50 * 0.9 * 50.0)
    # Storage: the central schedule at the national price and its weekly make-whole.
    assert month.loc["bat", "wholesale_revenue_gbp"] == pytest.approx(5 * 60 - 5 * 70)
    assert month.loc["bat", "balancing_settlement_gbp"] == pytest.approx(25.0)
    # Nothing paid in real time when the switch is off: owners bear it.
    off = _central_month_settlement(w, b, PRICE, central_month_units(_units(), "C_Jan"), {"AR4": 90.0}, None,
                                    roc_value_gbp_per_roc=50.0, fuel_rates=FUEL,
                                    realtime_at_cost=False).set_index("asset_id")
    assert off.loc["gas", "balancing_settlement_gbp"] == pytest.approx(1000.0)


def test_transfers_to_a_unit_the_network_does_not_carry_are_refused(tmp_path):
    w, b = _months(tmp_path)
    units = _units()
    units.loc[0, "asset_id"] = "ghost"
    with pytest.raises(ValueError, match="does not carry"):
        _central_month_settlement(w, b, PRICE, central_month_units(units, "C_Jan"), None, None, 0.0)
    with pytest.raises(ValueError, match="no rows"):
        central_month_units(_units(), "C_Feb")


def test_the_annual_settlement_books_the_lost_top_up_once(tmp_path, monkeypatch):
    w, b = _months(tmp_path)
    import scripts.public_ownership.settlement as st

    # The market months' Stage 1 hourly inputs (national price and demand) without a market network.
    monkeypatch.setattr(st, "read_hourly_inputs",
                        lambda network, price, links=None: pd.DataFrame({"price_gbp_per_mwh": PRICE,
                                                                         "demand_mw": 100.0, "hours": 1.0}))
    out = settle_assets_central_dispatch(["M_Jan"], ["C_Jan"], ["M_Jan_wholesale.nc"], [w], [b], ["p.csv"], _units(),
                                         strike_table={"AR4": 90.0}, reference_price_basis=None,
                                         roc_value_gbp_per_roc=50.0, marginal_cost_breakdown=None)
    assert list(out.columns) == SETTLEMENT_COLUMNS
    rows = out.set_index("asset_id")
    wind = rows.loc["wind"]
    assert wind["market_revenue_gbp"] == pytest.approx(wind["wholesale_revenue_gbp"] + 700.0)
    assert wind["support_revenue_gbp"] == pytest.approx(wind["cfd_support_gbp"])     # the 300 is not added again
    transfers = out.attrs["transfers"].set_index("asset_id")
    assert set(CENTRAL_TRANSFER_COLUMNS) <= set(transfers.columns)
    assert transfers.loc["bat", "storage_make_whole_gbp"] == 25.0
    assert transfers[list(CENTRAL_TRANSFER_COLUMNS)].sum(axis=1).to_dict() == pytest.approx(
        rows["balancing_settlement_gbp"].reindex(transfers.index).to_dict())
    with pytest.raises(ValueError, match="aligned"):
        settle_assets_central_dispatch(["M_Jan"], [], ["x"], [w], [b], ["p"], _units(), strike_table=None,
                                       reference_price_basis=None, roc_value_gbp_per_roc=0.0)
