"""FES 2050 plan, task Y14: hydrogen turbines burn hydrogen priced at their hydrogen bus.

From the vintaged years the hydrogen-turbine Links are register assets (Y14-prep), so
their settlement row must carry a fuel cost or their margin is the whole revenue.
``hydrogen_fuel_price="bus0_marginal_price"`` books it: AC-side balancing delivery /
efficiency x the Stage 1 marginal price of the link's bus0.  The default (None) books
nothing, which is every 2030 row.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from scripts.public_ownership.settlement import (
    FUEL_BASIS_HYDROGEN_BUS, FUEL_BASIS_NONE, HYDROGEN_FUEL_BUS0_PRICE, _hydrogen_link_rows,
    hydrogen_turbine_fuel_gbp, settle_assets,
)

SNAPSHOTS = pd.date_range("2050-01-01", periods=3, freq="h")
HOURS = pd.Series([1.0, 1.0, 2.0], index=SNAPSHOTS)
LINKS = ["H2T1", "H2T2", "EL1"]


def _datasets(h2_price=(40.0, 50.0, 60.0), p0_turbine=(10.0, 20.0, -1.0)):
    n = len(SNAPSHOTS)
    db = xr.Dataset(
        {
            "links_carrier": ("links_i", ["H2_turbine", "H2_turbine", "electrolysis"]),
            "links_bus0": ("links_i", ["H2_N", "H2_S", "AC_S"]),
            "links_bus1": ("links_i", ["AC_N", "AC_S", "H2_S"]),
            "links_efficiency": ("links_i", [0.5, 0.4, 0.7]),
            "buses_carrier": ("buses_i", ["AC", "AC", "H2", "H2"]),
            "links_t_p0": (("snapshots", "links_t_p0_i"),
                           np.column_stack([p0_turbine, [5.0, 5.0, 5.0], [30.0, 0.0, 10.0]])),
        },
        coords={"links_i": LINKS, "buses_i": ["AC_N", "AC_S", "H2_N", "H2_S"], "links_t_p0_i": LINKS,
                "snapshots": np.arange(n)},
    )
    dw = xr.Dataset(
        {"buses_t_marginal_price": (("snapshots", "buses_t_marginal_price_i"),
                                    np.column_stack([[70.0] * n, [70.0] * n, list(h2_price), [10.0, 20.0, 30.0]]))},
        coords={"snapshots": np.arange(n), "buses_t_marginal_price_i": ["AC_N", "AC_S", "H2_N", "H2_S"]},
    )
    return dw, db


def _rows(tmp_path, fuel, **kw):
    dw, db = _datasets(**kw)
    wholesale = tmp_path / "FES25_HT_2050_Jan_market_wholesale.nc"
    pd.DataFrame({"H2T1": [8.0, 8.0, 8.0], "H2T2": [5.0, 5.0, 5.0], "EL1": [0.0, 0.0, 0.0]},
                 index=SNAPSHOTS).to_csv(tmp_path / "FES25_HT_2050_Jan_market_wholesale_links.csv")
    coordinates = pd.DataFrame({"lat": [57.0, 52.0, 57.0, 52.0], "lon": [-3.0, -1.0, -3.0, -1.0]},
                               index=["AC_N", "AC_S", "H2_N", "H2_S"])
    redispatch = pd.DataFrame({"net_cost": [0.0, 0.0, 0.0]}, index=LINKS)
    price = pd.Series(70.0, index=SNAPSHOTS)
    return _hydrogen_link_rows(dw, db, wholesale, SNAPSHOTS, HOURS, price, redispatch, coordinates,
                               ("electrolysis", "H2_turbine"), hydrogen_fuel_price=fuel).set_index("asset_id")


def test_the_default_books_no_hydrogen_fuel(tmp_path):
    rows = _rows(tmp_path, None)
    assert rows["fuel_carbon_gbp"].eq(0.0).all()
    assert rows["fuel_carbon_basis"].eq(FUEL_BASIS_NONE).all()


def test_turbines_pay_their_hydrogen_bus_price_on_the_hydrogen_they_burn(tmp_path):
    rows = _rows(tmp_path, HYDROGEN_FUEL_BUS0_PRICE)
    # H2T1: AC delivery 5, 10, (clipped) 0 MW at efficiency 0.5 -> 10, 20, 0 MWh of H2
    # at 40, 50, 60 GBP/MWh over hours 1, 1, 2.
    assert rows.loc["H2T1", "fuel_carbon_gbp"] == pytest.approx(10 * 40 + 20 * 50)
    # H2T2: 5 MW of H2 every hour at its own bus (H2_S): 5 x (10 + 20 + 2 x 30).
    assert rows.loc["H2T2", "fuel_carbon_gbp"] == pytest.approx(5 * (10 + 20 + 60))
    assert rows.loc[["H2T1", "H2T2"], "fuel_carbon_basis"].eq(FUEL_BASIS_HYDROGEN_BUS).all()
    # electrolysers buy electricity (negative revenue), never fuel
    assert rows.loc["EL1", "fuel_carbon_gbp"] == 0.0 and rows.loc["EL1", "fuel_carbon_basis"] == FUEL_BASIS_NONE
    # the AC-side dispatch is the one the row already reports
    assert rows.loc["H2T1", "balancing_mwh"] == pytest.approx(5 + 10 - 0.5 * 2)
    margin = rows.loc["H2T1", "wholesale_revenue_gbp"] - rows.loc["H2T1", "fuel_carbon_gbp"]
    assert margin < rows.loc["H2T1", "wholesale_revenue_gbp"]


def test_a_missing_hydrogen_price_is_refused_not_zero_filled(tmp_path):
    dw, db = _datasets()
    dw = dw.isel(buses_t_marginal_price_i=[0, 1])
    pd.DataFrame({"H2T1": [8.0] * 3, "H2T2": [5.0] * 3, "EL1": [0.0] * 3},
                 index=SNAPSHOTS).to_csv(tmp_path / "X_wholesale_links.csv")
    with pytest.raises(ValueError, match="hydrogen bus price"):
        _hydrogen_link_rows(dw, db, tmp_path / "X_wholesale.nc", SNAPSHOTS, HOURS, pd.Series(70.0, index=SNAPSHOTS),
                            pd.DataFrame({"net_cost": [0.0] * 3}, index=LINKS),
                            pd.DataFrame({"lat": [0.0] * 4, "lon": [0.0] * 4},
                                         index=["AC_N", "AC_S", "H2_N", "H2_S"]),
                            ("electrolysis", "H2_turbine"), hydrogen_fuel_price=HYDROGEN_FUEL_BUS0_PRICE)


def test_the_fuel_function_refuses_bad_inputs():
    delivery = pd.DataFrame({"T": [1.0, 2.0]})
    price = pd.DataFrame({"H": [10.0, 10.0]})
    hours = pd.Series([1.0, 1.0])
    assert hydrogen_turbine_fuel_gbp(delivery, pd.Series({"T": 0.5}), pd.Series({"T": "H"}), [True], price,
                                     hours)["T"] == pytest.approx(60.0)
    with pytest.raises(ValueError, match="basis"):
        hydrogen_turbine_fuel_gbp(delivery, pd.Series({"T": 0.5}), pd.Series({"T": "H"}), [True], price, hours,
                                  basis="lcoh")
    with pytest.raises(ValueError, match="bus0"):
        hydrogen_turbine_fuel_gbp(delivery, pd.Series({"T": 0.5}), pd.Series({"T": "H"}), [False], price, hours)
    with pytest.raises(ValueError, match="efficiency"):
        hydrogen_turbine_fuel_gbp(delivery, pd.Series({"T": 0.0}), pd.Series({"T": "H"}), [True], price, hours)


def test_settle_assets_refuses_an_unknown_hydrogen_basis():
    with pytest.raises(ValueError, match="hydrogen_fuel_price"):
        settle_assets(["a"], ["w"], ["b"], ["p"], ["r"], strike_table=None, reference_price_basis=None,
                      roc_value_gbp_per_roc=0.0, hydrogen_fuel_price="lcoh")
