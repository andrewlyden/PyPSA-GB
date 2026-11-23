"""Item F8: state-owned storage and thermal plant bid at cost in the balancing stage.

Before F8 only the state's contract plant bid at cost (item BM1); its batteries, long-duration
storage and gas bid as private units would, so the margins in their bids and offers stayed a
transfer to the state's own plant. F8 blends each unit towards its at-cost price by its state
share, as the CfD bids already do, and is off unless a scenario switches it on.
"""
import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.market.cfd_bm_bids import apply_state_thermal_at_cost
from scripts.market.market_utils import _attach_storage_price_frames

LOG = logging.getLogger("test_state_at_cost_bids")
SNAPS = pd.date_range("2030-01-01", periods=3, freq="h")
GUARD = 250.0


def _shares_file(tmp_path, shares: dict) -> str:
    path = tmp_path / "shares.csv"
    pd.DataFrame([{"case_id": "p", "generator": g, "capacity_mw": 1.0, "state_capacity_mw": s, "state_share": s,
                   "state_commissioning_year": 2028} for g, s in shares.items()]).to_csv(path, index=False)
    return str(path)


def _balancing():
    return {"storage_bid_reference": "stage1_price", "storage_churn_guard_gbp_per_mwh": GUARD,
            "carrier_overrides": {"Battery": {"mode": "round_trip", "bid_margin": 44.0, "bid_price": 40.0,
                                              "offer_premium": 67.0}}}


def _bm_bids(shares_file=None, storage_at_cost=False):
    """A scenario's ``support.bm_bids`` block."""
    block = {"enabled": True, "case_id": "p", "shares_file": shares_file, "public_bid_price": 0.0,
             "storage_at_cost": storage_at_cost} if shares_file else {}
    return {"enabled": True, "contract_value_bids": {"enabled": True, "public_ownership_bids": block}}


def _storage_network():
    eta_leg = np.sqrt(0.846)
    su = pd.DataFrame({"carrier": ["Battery", "Battery", "Battery"], "efficiency_store": eta_leg,
                       "efficiency_dispatch": eta_leg}, index=["PRIVATE", "PUBLIC", "HALF"])
    net = SimpleNamespace(storage_units=su, snapshots=SNAPS)
    net._bm_reference_price = pd.Series([0.0, 80.0, 150.0], index=SNAPS)
    return net


def _static(su):
    eta = su["efficiency_store"] * su["efficiency_dispatch"]
    paid = 40.0
    return pd.Series(paid / eta + 67.0, index=su.index), pd.Series(-paid, index=su.index)


def test_storage_unchanged_when_the_switch_is_off(tmp_path):
    net = _storage_network()
    offer, bid = _static(net.storage_units)
    _attach_storage_price_frames(net, _balancing(), offer, bid, LOG, bm_bids=_bm_bids())
    base_offer, base_bid = net._bm_su_offer_tv.copy(), net._bm_su_bid_tv.copy()
    net2 = _storage_network()
    path = _shares_file(tmp_path, {"PUBLIC": 1.0, "HALF": 0.5})
    _attach_storage_price_frames(net2, _balancing(), offer, bid, LOG, bm_bids=_bm_bids(path, storage_at_cost=False))
    pd.testing.assert_frame_equal(net2._bm_su_offer_tv, base_offer)
    pd.testing.assert_frame_equal(net2._bm_su_bid_tv, base_bid)


def test_state_storage_drops_the_margin_but_keeps_the_churn_guard(tmp_path):
    net = _storage_network()
    offer, bid = _static(net.storage_units)
    path = _shares_file(tmp_path, {"PUBLIC": 1.0, "HALF": 0.5})
    _attach_storage_price_frames(net, _balancing(), offer, bid, LOG, bm_bids=_bm_bids(path, storage_at_cost=True))
    lam = net._bm_reference_price.to_numpy()
    eta = 0.846
    guard_premium = GUARD * (1 - eta) / eta
    # Private: pays the price less the 44 margin to charge, offers its round trip plus 67.
    assert net._bm_su_bid_tv["PRIVATE"].to_numpy() == pytest.approx(-(lam - 44.0))
    assert net._bm_su_offer_tv["PRIVATE"].to_numpy() == pytest.approx((lam - 44.0) / eta + 67.0)
    # Wholly state-owned: no margin, and only the churn guard's minimum premium.
    assert net._bm_su_bid_tv["PUBLIC"].to_numpy() == pytest.approx(-lam)
    assert net._bm_su_offer_tv["PUBLIC"].to_numpy() == pytest.approx(lam / eta + guard_premium)
    # Half state-owned: the capacity-weighted blend.
    assert net._bm_su_bid_tv["HALF"].to_numpy() == pytest.approx(-(lam - 22.0))
    assert net._bm_su_offer_tv["HALF"].to_numpy() == pytest.approx((lam - 22.0) / eta + 0.5 * 67.0 + 0.5 * guard_premium)
    # Churn still costs at least the guard per MWh destroyed for every unit.
    for unit in ("PRIVATE", "PUBLIC", "HALF"):
        churn = (net._bm_su_offer_tv[unit] * eta + net._bm_su_bid_tv[unit]) / (1 - eta)
        assert (churn >= GUARD - 1e-6).all()


def _thermal(shares=None):
    gens = pd.DataFrame({"carrier": ["CCGT", "CCGT", "OCGT", "wind_offshore"],
                         "marginal_cost": [110.0, 120.0, 160.0, 0.0]},
                        index=["CCGT_PRIV", "CCGT_STATE", "OCGT_HALF", "WIND"])
    net = SimpleNamespace(generators=gens)
    offer = gens["marginal_cost"] * 1.29
    # ESO-cost convention, net_saving: NESO receives the fuel a unit saves by turning down.
    bid = -gens["marginal_cost"] * 0.9
    offer_tv = pd.DataFrame(np.tile(offer.to_numpy(), (len(SNAPS), 1)), index=SNAPS, columns=gens.index)
    bid_tv = pd.DataFrame(np.tile(bid.to_numpy(), (len(SNAPS), 1)), index=SNAPS, columns=gens.index)
    return net, offer, bid, offer_tv, bid_tv


def test_state_thermal_offers_and_bids_at_short_run_cost():
    net, offer, bid, offer_tv, bid_tv = _thermal()
    shares = pd.Series({"CCGT_STATE": 1.0, "OCGT_HALF": 0.5, "WIND": 1.0})
    carrier = net.generators["carrier"].str.lower()
    moved = apply_state_thermal_at_cost(net, {"thermal_at_cost": True}, shares, offer_tv, bid_tv, offer, bid,
                                        carrier, LOG)
    assert moved == 2
    assert offer_tv["CCGT_STATE"].to_numpy() == pytest.approx(120.0)
    # The state hands its whole avoided cost back (13 Oct 2026: it was +mc, so NESO paid it to turn down).
    assert bid_tv["CCGT_STATE"].to_numpy() == pytest.approx(-120.0)
    assert offer_tv["OCGT_HALF"].to_numpy() == pytest.approx(0.5 * 160.0 * 1.29 + 0.5 * 160.0)
    assert bid_tv["OCGT_HALF"].to_numpy() == pytest.approx(-0.5 * 160.0 * 0.9 - 0.5 * 160.0)
    assert bid_tv["CCGT_PRIV"].to_numpy() == pytest.approx(-110.0 * 0.9)
    # Private plant and non-thermal carriers are untouched.
    assert offer_tv["CCGT_PRIV"].to_numpy() == pytest.approx(110.0 * 1.29)
    assert offer_tv["WIND"].to_numpy() == pytest.approx(0.0)


def test_thermal_unchanged_when_the_switch_is_off():
    net, offer, bid, offer_tv, bid_tv = _thermal()
    before = offer_tv.copy()
    shares = pd.Series({"CCGT_STATE": 1.0})
    assert apply_state_thermal_at_cost(net, {}, shares, offer_tv, bid_tv, offer, bid,
                                       net.generators["carrier"].str.lower(), LOG) == 0
    pd.testing.assert_frame_equal(offer_tv, before)


def test_state_thermal_bid_keeps_the_legacy_sign_only_under_legacy_cost():
    net, offer, bid, offer_tv, bid_tv = _thermal()
    shares = pd.Series({"CCGT_STATE": 1.0})
    apply_state_thermal_at_cost(net, {"thermal_at_cost": True}, shares, offer_tv, bid_tv, offer, bid,
                                net.generators["carrier"].str.lower(), LOG, turn_down_saving=False)
    assert bid_tv["CCGT_STATE"].to_numpy() == pytest.approx(120.0)


def _renewables():
    names = ["ROC_ON", "ROC_HALF", "MERCH_PV", "CFD_OFF", "ROC_BIO", "ROC_PRIV"]
    support = pd.Series(["roc", "roc", "merchant", "cfd", "roc", "roc"], index=names)
    carrier = pd.Series(["wind_onshore", "wind_onshore", "solar_pv", "wind_offshore", "biomass", "wind_onshore"],
                        index=names)
    excluded = carrier.eq("biomass")
    private = pd.Series([72.0, 72.0, 20.0, 50.0, 30.0, 72.0], index=names)
    bid_tv = pd.DataFrame(np.tile(private.to_numpy(), (len(SNAPS), 1)), index=SNAPS, columns=names)
    shares = pd.Series({"ROC_ON": 1.0, "ROC_HALF": 0.5, "MERCH_PV": 1.0, "CFD_OFF": 1.0, "ROC_BIO": 1.0})
    return support, carrier, excluded, bid_tv, shares


def test_state_owned_ro_and_merchant_renewables_bid_the_public_price():
    from scripts.market.cfd_bm_bids import apply_state_renewables_at_cost
    support, carrier, excluded, bid_tv, shares = _renewables()
    moved = apply_state_renewables_at_cost({"renewables_at_cost": True}, shares, bid_tv, support, carrier,
                                           excluded, 0.0, LOG)
    assert set(moved.index) == {"ROC_ON", "ROC_HALF", "MERCH_PV"}
    assert bid_tv["ROC_ON"].to_numpy() == pytest.approx(0.0)
    assert bid_tv["ROC_HALF"].to_numpy() == pytest.approx(36.0)
    assert bid_tv["MERCH_PV"].to_numpy() == pytest.approx(0.0)
    # CfD units are priced by the CfD loop, excluded carriers and private units are left alone.
    assert bid_tv["CFD_OFF"].to_numpy() == pytest.approx(50.0)
    assert bid_tv["ROC_BIO"].to_numpy() == pytest.approx(30.0)
    assert bid_tv["ROC_PRIV"].to_numpy() == pytest.approx(72.0)


def test_ro_and_merchant_renewables_unchanged_when_the_switch_is_off():
    from scripts.market.cfd_bm_bids import apply_state_renewables_at_cost
    support, carrier, excluded, bid_tv, shares = _renewables()
    before = bid_tv.copy()
    assert apply_state_renewables_at_cost({}, shares, bid_tv, support, carrier, excluded, 0.0, LOG).empty
    pd.testing.assert_frame_equal(bid_tv, before)
