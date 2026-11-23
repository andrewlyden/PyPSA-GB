"""F18 (27 Oct 2026): two bill reports that change no bill line.

* The constraint line split into thermal redispatch and interconnector re-trading, beside
  NESO's thermal constraint cost for the case's network.
* The share of CfD difference payments made in hours at or below a zero (and GBP 1) price.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from scripts.consumer_bill import project_future_bill as bill
from scripts.market.revenue_tracking import compute_cfd_payments


# ── the constraint line beside NESO's thermal constraint cost ────────────────

def test_the_constraint_line_split_shows_retrading_apart_and_keeps_the_total():
    balancing = {"constraint_cost_gbp": 4.0e9, "interconnector_retrading_cost_gbp": 0.5e9,
                 "include_interconnector_retrading": True}
    case = {"tnuos_network_delivery": {"file": "x.json", "case": "cp30"}}
    rate = 3.1 / 3.0e8
    split = bill.constraint_line_split(balancing, case, household_per_system=rate)
    assert split["constraint_line_gbp"] == 4.0e9
    assert split["thermal_redispatch_gbp"] + split["interconnector_retrading_gbp"] == pytest.approx(4.0e9)
    assert split["thermal_redispatch_gbp"] == pytest.approx(3.5e9)
    assert split["constraint_line_gbp_per_3100kwh_ex_vat"] == pytest.approx(4.0e9 * rate)
    neso = split["neso_thermal_constraint_cost"]
    assert neso["value_GBP"] == pytest.approx(3.57e9) and neso["network_case"] == "cp30"
    assert neso["thermal_redispatch_less_neso_gbp"] == pytest.approx(3.5e9 - 3.57e9)
    assert neso["constraint_line_less_neso_gbp"] == pytest.approx(4.0e9 - 3.57e9)
    # Each network case reads its own NESO figure; a case naming none gets the split alone.
    central = bill.constraint_line_split(balancing, {"tnuos_network_delivery": {"case": "no_acceleration"}},
                                         household_per_system=rate)
    assert central["neso_thermal_constraint_cost"]["value_GBP"] == pytest.approx(7.79e9)
    assert "neso_thermal_constraint_cost" not in bill.constraint_line_split(balancing, {}, household_per_system=rate)
    with pytest.raises(ValueError, match="no NESO constraint benchmark"):
        bill.constraint_line_split(balancing, {"tnuos_network_delivery": {"case": "seven_late_x"}},
                                   household_per_system=rate)
    # Re-trading outside the line is not taken off the thermal part.
    outside = bill.constraint_line_split({**balancing, "include_interconnector_retrading": False}, {},
                                         household_per_system=rate)
    assert outside["thermal_redispatch_gbp"] == 4.0e9 and outside["interconnector_retrading_outside_line_gbp"] == 0.5e9


# ── CfD payments in zero-price hours ─────────────────────────────────────────

def _network(dispatch, rounds=("AR4", "AR4")):
    snapshots = dispatch.index
    generators = pd.DataFrame({"support_type": "CfD", "carrier": "wind_offshore", "bus": "GB",
                               "cfd_round": list(rounds)}, index=dispatch.columns)
    return SimpleNamespace(generators=generators, snapshots=snapshots,
                           generators_t=SimpleNamespace(p=dispatch),
                           snapshot_weightings=pd.DataFrame({"generators": 1.0}, index=snapshots))


def test_the_price_bands_split_each_payment_without_changing_it():
    index = pd.date_range("2030-01-02", periods=4, freq="h")
    dispatch = pd.DataFrame({"a": [10.0, 10.0, 10.0, 10.0], "b": [0.0, 5.0, 5.0, 5.0]}, index=index)
    price = pd.Series([-0.005, 0.5, 40.0, 80.0], index=index)
    bands = {"le0": 0.0, "le1": 1.0}
    plain = compute_cfd_payments(_network(dispatch), {"AR4_offshore": 60.0}, reference_price=price)
    banded = compute_cfd_payments(_network(dispatch), {"AR4_offshore": 60.0}, reference_price=price, price_bands=bands)
    pd.testing.assert_series_equal(plain["net_cfd_cost"], banded["net_cfd_cost"])
    a = banded.set_index("generator").loc["a"]
    assert a["cfd_payment_le0"] == pytest.approx(10.0 * 60.005)
    assert a["cfd_payment_le1"] == pytest.approx(10.0 * 60.005 + 10.0 * 59.5)
    assert a["dispatch_MWh_le1"] == pytest.approx(20.0)
    # The negative-price rule applies first: no payment in a negative hour, so none in its band.
    ruled = compute_cfd_payments(_network(dispatch), {"AR4_offshore": 60.0}, reference_price=price,
                                 negative_price_rule="ar4_zero_payment_when_price_le_0", price_bands=bands)
    assert ruled.set_index("generator").loc["a", "cfd_payment_le0"] == pytest.approx(0.0)


def test_the_bill_records_the_share_of_the_cfd_line_in_low_price_hours(tmp_path):
    snapshots = pd.date_range("2030-01-02", periods=3, freq="h")
    ds = xr.Dataset({
        "snapshots_snapshot": ("snapshots", snapshots), "snapshots_generators": ("snapshots", [1., 1., 1.]),
        "generators_support_type": ("generators_i", ["CfD"]), "generators_carrier": ("generators_i", ["wind_offshore"]),
        "generators_bus": ("generators_i", ["GB"]), "generators_cfd_round": ("generators_i", ["AR8_modelled"]),
        "generators_p_nom": ("generators_i", [20.]),
        "generators_t_p": (("snapshots", "generators_t_p_i"), [[10.], [10.], [10.]]),
    }, coords={"generators_i": ["wind"], "generators_t_p_i": ["wind"]})
    path = tmp_path / "physical.nc"
    ds.to_netcdf(path)
    price = pd.Series([0.0, 0.8, 50.0], index=snapshots)
    payment, *_, strikes = bill.read_cfd_payments(path, price, {"future_offshore": 60})
    assert payment == pytest.approx(10 * 60 + 10 * 59.2 + 10 * 10)
    low = strikes["low_price_hours"]
    assert low["price_at_or_below_0"]["payment_gbp"] == pytest.approx(600.0)
    assert low["price_at_or_below_1"]["payment_gbp"] == pytest.approx(600.0 + 592.0)
    hourly = pd.DataFrame({"price_gbp_per_mwh": price, "hours": 1.0})
    record = bill.cfd_low_price_record(low, hourly, payment, 50.0, 0.0)
    band = record["bands"]["price_at_or_below_1"]
    assert band["hours"] == 2.0
    assert band["share_of_settled_payment"] == pytest.approx(1192.0 / 1292.0)
    assert band["share_of_cfd_line"] == pytest.approx((1192.0 + 50.0) / 1342.0)
    with_hh = bill._with_household(record, 2.0)
    assert with_hh["bands"]["price_at_or_below_0"]["settled_payment_gbp_per_3100kwh_ex_vat"] == pytest.approx(1200.0)
    assert "settled_payment_gbp_per_3100kwh_ex_vat" not in record["bands"]["price_at_or_below_0"]
    assert np.isclose(with_hh["cfd_line_gbp_per_3100kwh_ex_vat"], 2.0 * 1342.0)
