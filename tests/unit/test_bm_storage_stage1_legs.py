"""BM storage anchored on Stage 1's gross legs (market.balancing.storage_anchor, 10 Oct 2026).

With a calibrated churn share Stage 1 charges and discharges a unit at once. Net p
hides that, and a BM anchored on the net split by sign starts every hour without the
churn: on the CP30 2030 churn_neso months it sold 5.5 TWh of energy Stage 1 had burnt
as cheap storage offers. These tests pin the stage1_legs path: the legs are read back
from Stage 1's saved network, the rebuilt state of charge is exact, and a BM with no
redispatch to do reproduces Stage 1's churn instead of undoing it.
"""

import logging

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.market.solve_balancing import (
    _build_balancing_extra_functionality,
    _storage_reverse_leg_cost,
    read_stage1_storage_legs,
    resolve_storage_anchor,
    stage1_state_of_charge,
    stage1_storage_leg_arrays,
)

ETA = 0.9
LOG = logging.getLogger("test_bm_stage1_legs")


def _network(line_s_nom=1000.0):
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=4, freq="h"))
    for carrier in ("AC", "wind_onshore", "CCGT", "Battery"):
        n.add("Carrier", carrier)
    n.add("Bus", "A", v_nom=400, carrier="AC")
    n.add("Bus", "B", v_nom=400, carrier="AC")
    n.add("Line", "A_B", bus0="A", bus1="B", r=0.01, x=0.1, s_nom=line_s_nom, carrier="AC")
    n.add("Generator", "cheap", bus="A", p_nom=300.0, carrier="wind_onshore", marginal_cost=10.0)
    n.add("Generator", "dear", bus="B", p_nom=300.0, carrier="CCGT", marginal_cost=100.0)
    n.add("StorageUnit", "battery", bus="B", p_nom=50.0, max_hours=4.0, carrier="Battery",
          marginal_cost=1.0, efficiency_store=ETA, efficiency_dispatch=ETA,
          state_of_charge_initial=100.0, cyclic_state_of_charge=False, standing_loss=0.0)
    n.add("Load", "load", bus="B", p_set=150.0)
    return n


def _churn_legs(snaps):
    """Stage 1 churns: 20 MW in and 10 MW out every hour (net -10)."""
    return (pd.DataFrame({"battery": 20.0}, index=snaps), pd.DataFrame({"battery": 10.0}, index=snaps))


@pytest.mark.unit
class TestLegArrays:

    def test_net_sign_split_without_legs(self):
        snaps = pd.date_range("2030-01-01", periods=2, freq="h")
        su = pd.DataFrame({"battery": [-10.0, 5.0]}, index=snaps)
        d, s = stage1_storage_leg_arrays(su, ["battery"], snaps)
        assert d.ravel().tolist() == [0.0, 5.0]
        assert s.ravel().tolist() == [10.0, 0.0]

    def test_legs_used_when_given(self):
        snaps = pd.date_range("2030-01-01", periods=2, freq="h")
        su = pd.DataFrame({"battery": -10.0}, index=snaps)
        d, s = stage1_storage_leg_arrays(su, ["battery"], snaps, _churn_legs(snaps))
        assert d.ravel().tolist() == [10.0, 10.0]
        assert s.ravel().tolist() == [20.0, 20.0]

    def test_anchor_setting_validated(self):
        assert resolve_storage_anchor({}) == "net_sign"
        assert resolve_storage_anchor({"storage_anchor": "stage1_legs"}) == "stage1_legs"
        with pytest.raises(ValueError):
            resolve_storage_anchor({"storage_anchor": "gross"})


@pytest.mark.unit
class TestReadLegs:

    def test_round_trip_through_netcdf(self, tmp_path):
        n = _network()
        store, dispatch = _churn_legs(n.snapshots)
        n.storage_units_t.p_store = store
        n.storage_units_t.p_dispatch = dispatch
        n.storage_units_t.p = dispatch - store
        path = tmp_path / "stage1.nc"
        n.export_to_netcdf(path)
        su = pd.DataFrame({"battery": -10.0}, index=n.snapshots)
        got_store, got_dispatch = read_stage1_storage_legs(path, su, LOG)
        pd.testing.assert_frame_equal(got_store, store, check_freq=False, check_names=False)
        pd.testing.assert_frame_equal(got_dispatch, dispatch, check_freq=False, check_names=False)

    def test_refuses_a_stage1_without_legs(self, tmp_path):
        n = _network()
        n.storage_units_t.p = pd.DataFrame({"battery": -10.0}, index=n.snapshots)
        path = tmp_path / "old_stage1.nc"
        n.export_to_netcdf(path)
        with pytest.raises(ValueError, match="no storage legs"):
            read_stage1_storage_legs(path, n.storage_units_t.p, LOG)

    def test_refuses_legs_that_do_not_net_to_the_schedule(self, tmp_path):
        n = _network()
        store, dispatch = _churn_legs(n.snapshots)
        n.storage_units_t.p_store = store
        n.storage_units_t.p_dispatch = dispatch
        path = tmp_path / "stage1.nc"
        n.export_to_netcdf(path)
        su = pd.DataFrame({"battery": -5.0}, index=n.snapshots)
        with pytest.raises(ValueError, match="do not net"):
            read_stage1_storage_legs(path, su, LOG)


@pytest.mark.unit
class TestStateOfCharge:

    def test_legs_give_the_exact_path(self):
        n = _network()
        snaps = n.snapshots
        su = pd.DataFrame({"battery": -10.0}, index=snaps)
        exact = stage1_state_of_charge(n, su, legs=_churn_legs(snaps))["battery"]
        step = 20.0 * ETA - 10.0 / ETA
        assert exact.tolist() == pytest.approx([100.0 + step * (k + 1) for k in range(4)])
        # The net path (no churn) overstates the state of charge.
        net = stage1_state_of_charge(n, su)["battery"]
        assert (net > exact + 1.0).all()


@pytest.mark.unit
class TestBalancingKeepsStage1Churn:

    def _solve(self, legs):
        n = _network()
        snaps = n.snapshots
        gen = pd.DataFrame({"cheap": 160.0, "dear": 0.0}, index=snaps)
        su = pd.DataFrame({"battery": -10.0}, index=snaps)
        extra = _build_balancing_extra_functionality(
            wholesale_gen=gen, wholesale_su=su, wholesale_links=pd.DataFrame(index=snaps),
            gen_offer_prices=pd.Series({"cheap": 12.0, "dear": 110.0}),
            gen_bid_prices=pd.Series({"cheap": 9.0, "dear": 90.0}),
            su_offer_prices=pd.Series({"battery": 59.0}),
            su_bid_prices=pd.Series({"battery": 50.0}),
            fix_interconnectors=False, logger=LOG, wholesale_su_legs=legs,
        )
        status, condition = n.optimize(solver_name="highs", extra_functionality=extra)
        assert status == "ok", condition
        return n, su

    def test_uncongested_bm_reproduces_the_churn(self):
        n, su = self._solve(_churn_legs(_network().snapshots))
        assert n.storage_units_t.p_store["battery"].tolist() == pytest.approx([20.0] * 4, abs=1e-6)
        assert n.storage_units_t.p_dispatch["battery"].tolist() == pytest.approx([10.0] * 4, abs=1e-6)
        inc = n.model.variables["Generator-increase"].solution.to_pandas().to_numpy()
        dec = n.model.variables["Generator-decrease"].solution.to_pandas().to_numpy()
        assert np.abs(inc).max() < 1e-6 and np.abs(dec).max() < 1e-6

    def test_net_sign_anchor_starts_without_the_churn(self):
        n, _ = self._solve(None)
        assert n.storage_units_t.p_dispatch["battery"].abs().max() < 1e-6
        assert n.storage_units_t.p_store["battery"].tolist() == pytest.approx([10.0] * 4, abs=1e-6)

    def test_held_nonparticipant_short_of_charge_stays_feasible(self):
        """3 Nov 2026: a held consumer-flex unit whose carried state of charge fell below Stage 1's (solver
        tolerance over the rolling windows) moves its store leg at a penalty instead of failing the window."""
        from scripts.market.solve_balancing import consumer_flex_rescheduled, held_link_slack
        n = _network()
        snaps = n.snapshots
        n.add("Carrier", "EV_smart_charging")
        n.add("StorageUnit", "ev", bus="B", p_nom=2.0, max_hours=1.0, carrier="EV_smart_charging", p_max_pu=0.0,
              p_min_pu=-1.0, efficiency_store=0.9, efficiency_dispatch=1.0, state_of_charge_initial=0.0,
              cyclic_state_of_charge=False, inflow=pd.Series(-0.9, index=snaps))
        gen = pd.DataFrame({"cheap": 160.0, "dear": 0.0}, index=snaps)
        # Stage 1 charges 0.98 MW (0.882 MWh in) against a 0.9 MWh drain: it needed charge this BM run lacks.
        su = pd.DataFrame({"battery": -10.0, "ev": -0.98}, index=snaps)
        extra = _build_balancing_extra_functionality(
            wholesale_gen=gen, wholesale_su=su, wholesale_links=pd.DataFrame(index=snaps),
            gen_offer_prices=pd.Series({"cheap": 12.0, "dear": 110.0}),
            gen_bid_prices=pd.Series({"cheap": 9.0, "dear": 90.0}),
            su_offer_prices=pd.Series({"battery": 59.0, "ev": 59.0}),
            su_bid_prices=pd.Series({"battery": 50.0, "ev": 50.0}),
            fix_interconnectors=False, logger=LOG,
            participating_storage_units=["battery"], fixed_storage_units=["ev"],
        )
        status, condition = n.optimize(solver_name="highs", extra_functionality=extra)
        assert status == "ok", condition
        # Only the shortfall moves (0.02 MW an hour, the solver may bring it forward), and the unit never runs dry.
        assert n.storage_units_t.p_store["ev"].sum() == pytest.approx(4.0, abs=1e-6)
        assert n.storage_units_t.state_of_charge["ev"].min() >= -1e-9
        # Consumer flexibility moves at its own price and is reported apart from the held-link slack.
        cost, mwh = consumer_flex_rescheduled(n)
        assert mwh == pytest.approx(0.02 * 4, abs=1e-6) and cost == pytest.approx(mwh * 1_000.0)
        assert held_link_slack(n) == (0.0, 0.0)

    @pytest.mark.parametrize("carrier, reschedules", [("SH_flexibility", True), ("Battery", False)])
    def test_held_flex_charging_rescheduled_before_load_is_shed(self, carrier, reschedules):
        """4 Nov 2026 (London, Mar/Apr 2030): Stage 1 schedules held flexible heating to charge behind an import
        limit. Consumer flexibility is rescheduled (1,000 GBP/MWh) before load is shed (VoLL); other held
        storage keeps its 10,000 GBP/MWh hold, so load is shed instead."""
        from scripts.market.solve_balancing import consumer_flex_rescheduled
        n = _network(line_s_nom=50.0)
        snaps = n.snapshots
        n.generators.at["dear", "p_nom"] = 100.0
        if carrier not in n.carriers.index:
            n.add("Carrier", carrier)
        n.add("Carrier", "load_shedding")
        n.add("Generator", "shed", bus="B", p_nom=1000.0, carrier="load_shedding", marginal_cost=6000.0)
        n.add("StorageUnit", "held", bus="B", p_nom=200.0, max_hours=4.0, carrier=carrier, p_max_pu=0.0,
              p_min_pu=-1.0, efficiency_store=1.0, efficiency_dispatch=1.0, state_of_charge_initial=0.0,
              cyclic_state_of_charge=False)
        gen = pd.DataFrame({"cheap": 160.0, "dear": 0.0, "shed": 0.0}, index=snaps)
        su = pd.DataFrame({"battery": -10.0, "held": -200.0}, index=snaps)
        extra = _build_balancing_extra_functionality(
            wholesale_gen=gen, wholesale_su=su, wholesale_links=pd.DataFrame(index=snaps),
            gen_offer_prices=pd.Series({"cheap": 12.0, "dear": 110.0, "shed": 6000.0}),
            gen_bid_prices=pd.Series({"cheap": 9.0, "dear": 90.0, "shed": 0.0}),
            su_offer_prices=pd.Series({"battery": 59.0, "held": 59.0}),
            su_bid_prices=pd.Series({"battery": 50.0, "held": 50.0}),
            fix_interconnectors=False, logger=LOG,
            participating_storage_units=["battery"], fixed_storage_units=["held"],
        )
        status, condition = n.optimize(solver_name="highs", extra_functionality=extra)
        assert status == "ok", condition
        shed = n.generators_t.p["shed"].sum()
        moved = 800.0 - n.storage_units_t.p_store["held"].sum()
        _, flex_mwh = consumer_flex_rescheduled(n)
        if reschedules:
            assert shed == pytest.approx(0.0, abs=1e-6) and moved > 1.0
            assert flex_mwh == pytest.approx(moved, abs=1e-6)
        else:
            assert shed > 1.0 and moved == pytest.approx(0.0, abs=1e-6) and flex_mwh == 0.0

    def test_reverse_leg_cost_counts_only_increases_over_the_legs(self):
        n, su = self._solve(_churn_legs(_network().snapshots))
        prices = pd.Series({"battery": 30.0})
        assert _storage_reverse_leg_cost(n, su, prices, _churn_legs(n.snapshots)) == pytest.approx(0.0, abs=1e-6)
        # On the net-sign split the kept 10 MW discharge would count as a reverse leg.
        assert _storage_reverse_leg_cost(n, su, prices) == pytest.approx(30.0 * 10.0 * 4, rel=1e-6)
