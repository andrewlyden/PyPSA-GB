"""Storage energy conservation in the balancing-mechanism objective.

The BM replaces PyPSA's objective with the redispatch cost, and anchors storage
on net ``p_dispatch - p_store``. Until gross ``p_dispatch`` was priced, the split
behind any net position was free and a barrier solve without crossover charged
and discharged the same unit at once, burning the round-trip loss (3.6 TWh
across the 2030 months). These tests pin down the structure of the fix rather
than the solver's behaviour: the term is in the objective, the reported BM cost
excludes it, and the solved storage dispatch conserves energy.
"""

import logging

import pandas as pd
import pypsa
import pytest

from scripts.market.solve_balancing import (
    _build_balancing_extra_functionality,
    _log_storage_energy_balance,
    _storage_end_soc_shortfall,
    _storage_energy_cost,
    stage1_state_of_charge,
)
from config.config_loader import load_defaults

STORAGE_MC = 1.0        # the floored dispatch cost carried into the objective
ETA = 0.9               # per leg
SOC_INITIAL = 100.0     # MWh
BATTERY_OFFER = 59.0
BATTERY_BID = 50.0
CHEAP_OFFER, CHEAP_BID = 12.0, 9.0
DEAR_OFFER, DEAR_BID = 110.0, 90.0


@pytest.fixture
def constrained_network():
    """Two buses joined by a 100 MW line; the wholesale position overloads it.

    Cheap generation and the battery sit at A, the load and dear generation
    at B. The copperplate wholesale stage served the 150 MW load entirely from
    the cheap generator, so the BM must turn it down by 50 MW and find that
    energy on the B side - from the battery (offer 59) while it has charge,
    then from the dear generator (offer 110).
    """
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=4, freq="h"))
    for carrier in ("AC", "wind_onshore", "CCGT", "Battery"):
        n.add("Carrier", carrier)
    n.add("Bus", "A", v_nom=400, carrier="AC")
    n.add("Bus", "B", v_nom=400, carrier="AC")
    n.add("Line", "A_B", bus0="A", bus1="B", r=0.01, x=0.1, s_nom=100.0, carrier="AC")
    n.add("Generator", "cheap", bus="A", p_nom=300.0, carrier="wind_onshore",
          marginal_cost=10.0)
    n.add("Generator", "dear", bus="B", p_nom=300.0, carrier="CCGT",
          marginal_cost=100.0)
    n.add("StorageUnit", "battery", bus="B", p_nom=50.0, max_hours=4.0,
          carrier="Battery", marginal_cost=STORAGE_MC,
          efficiency_store=ETA, efficiency_dispatch=ETA,
          state_of_charge_initial=SOC_INITIAL, cyclic_state_of_charge=False,
          standing_loss=0.0)
    n.add("Load", "load", bus="B", p_set=150.0)
    return n


def _wholesale_positions(n):
    snaps = n.snapshots
    gen = pd.DataFrame({"cheap": 150.0, "dear": 0.0}, index=snaps)
    su = pd.DataFrame({"battery": 0.0}, index=snaps)
    links = pd.DataFrame(index=snaps)
    return gen, su, links


def _solve_bm(n):
    gen, su, links = _wholesale_positions(n)
    extra = _build_balancing_extra_functionality(
        wholesale_gen=gen,
        wholesale_su=su,
        wholesale_links=links,
        gen_offer_prices=pd.Series({"cheap": CHEAP_OFFER, "dear": DEAR_OFFER}),
        gen_bid_prices=pd.Series({"cheap": CHEAP_BID, "dear": DEAR_BID}),
        su_offer_prices=pd.Series({"battery": BATTERY_OFFER}),
        su_bid_prices=pd.Series({"battery": BATTERY_BID}),
        fix_interconnectors=False,
        logger=logging.getLogger("test_bm_storage"),
    )
    status, condition = n.optimize(solver_name="highs", extra_functionality=extra)
    assert status == "ok", condition
    return n


@pytest.mark.unit
class TestStorageTermInObjective:

    def test_gross_dispatch_is_priced(self, constrained_network):
        """The objective carries marginal_cost on StorageUnit-p_dispatch."""
        n = _solve_bm(constrained_network)
        in_objective = set(
            int(label) for label in n.model.objective.expression.vars.values.ravel()
            if label >= 0
        )

        def labels_of(name):
            return set(int(v) for v in n.model.variables[name].labels.values.ravel())

        assert in_objective & labels_of("StorageUnit-p_dispatch")
        # and the redispatch variables are still there
        assert in_objective & labels_of("Generator-increase")
        assert in_objective & labels_of("Generator-decrease")

    def test_the_bm_really_redispatched(self, constrained_network):
        """Guard against the fixture degenerating into a no-op."""
        n = _solve_bm(constrained_network)
        assert n.generators_t.p["cheap"].max() <= 100.0 + 1e-6
        assert storage_dispatch_total(n) > 0.0


def storage_dispatch_total(n):
    return float(n.storage_units_t.p_dispatch["battery"].sum())


@pytest.mark.unit
class TestEnergyConservation:

    def test_no_simultaneous_charge_and_discharge(self, constrained_network):
        n = _solve_bm(constrained_network)
        overlap = (n.storage_units_t.p_dispatch["battery"]
                   * n.storage_units_t.p_store["battery"])
        assert overlap.abs().max() < 1e-6

    def test_state_of_charge_closes(self, constrained_network):
        """Stored minus drawn energy equals the change in state of charge."""
        n = _solve_bm(constrained_network)
        stored = n.storage_units_t.p_store["battery"].sum() * ETA
        drawn = n.storage_units_t.p_dispatch["battery"].sum() / ETA
        soc_final = n.storage_units_t.state_of_charge["battery"].iloc[-1]
        assert stored - drawn == pytest.approx(soc_final - SOC_INITIAL, abs=1e-6)

    def test_balance_check_reports_clean(self, constrained_network, caplog):
        n = _solve_bm(constrained_network)
        with caplog.at_level(logging.INFO, logger="test_bm_storage"):
            result = _log_storage_energy_balance(
                n, logging.getLogger("test_bm_storage"))
        assert result["simultaneous_unit_hours"] == 0
        assert result["burn_MWh"] == pytest.approx(0.0, abs=1e-6)
        ours = [r for r in caplog.records if r.name == "test_bm_storage"]
        assert ours, "the balance check should have logged its verdict"
        assert not any(r.levelno >= logging.WARNING for r in ours)

    def test_balance_check_flags_a_hidden_pair(self, constrained_network, caplog):
        """A simultaneous pair is invisible in net p; the check reads the gross
        legs (decision D6) and reports the energy it destroyed."""
        n = constrained_network
        n.storage_units_t.p = pd.DataFrame({"battery": [0.0] * 4}, index=n.snapshots)
        n.storage_units_t.p_dispatch = pd.DataFrame({"battery": [50.0] * 4}, index=n.snapshots)
        n.storage_units_t.p_store = pd.DataFrame({"battery": [50.0] * 4}, index=n.snapshots)
        logger = logging.getLogger("test_bm_storage_flag")
        with caplog.at_level(logging.WARNING, logger=logger.name):
            result = _log_storage_energy_balance(n, logger)
        eta_s = float(n.storage_units.at["battery", "efficiency_store"])
        eta_d = float(n.storage_units.at["battery", "efficiency_dispatch"])
        assert result["simultaneous_unit_hours"] == 4
        assert result["burn_MWh"] == pytest.approx(4 * 50.0 * (1 / eta_d - eta_s))
        assert any("curtailed energy" in r.getMessage() for r in caplog.records)


WIND_BID = 90.0


@pytest.fixture
def surplus_network():
    """A full battery at a bus with 50 MW of stranded surplus.

    Wind at A is anchored at 150 MW but the line to the load at B carries 100.
    Curtailing the 50 MW costs the wind bid (90 GBP/MWh). The battery at A is
    full, so it cannot absorb the surplus by charging - but under the old
    net-only anchor it could charge and discharge at once, burning
    (1/eta_d - eta_s) of the throughput as a sink priced only at the storage
    dispatch floor. That was the cheapest option, so the BM took it. With both
    legs anchored the pair pays offer + bid and curtailment wins.
    """
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=4, freq="h"))
    for carrier in ("AC", "wind_offshore", "CCGT", "Battery"):
        n.add("Carrier", carrier)
    n.add("Bus", "A", v_nom=400, carrier="AC")
    n.add("Bus", "B", v_nom=400, carrier="AC")
    n.add("Line", "A_B", bus0="A", bus1="B", r=0.01, x=0.1, s_nom=100.0, carrier="AC")
    n.add("Generator", "wind", bus="A", p_nom=200.0, carrier="wind_offshore",
          marginal_cost=0.5)
    n.add("Generator", "dear", bus="B", p_nom=300.0, carrier="CCGT",
          marginal_cost=100.0)
    n.add("StorageUnit", "battery", bus="A", p_nom=60.0, max_hours=4.0,
          carrier="Battery", marginal_cost=STORAGE_MC,
          efficiency_store=ETA, efficiency_dispatch=ETA,
          state_of_charge_initial=240.0, cyclic_state_of_charge=False,
          standing_loss=0.0)
    n.add("Load", "load", bus="B", p_set=150.0)
    return n


@pytest.mark.unit
class TestChurnIsNotADumpLoad:

    def test_surplus_is_curtailed_not_burnt_in_storage(self, surplus_network):
        n = surplus_network
        snaps = n.snapshots
        extra = _build_balancing_extra_functionality(
            wholesale_gen=pd.DataFrame({"wind": 150.0, "dear": 0.0}, index=snaps),
            wholesale_su=pd.DataFrame({"battery": 0.0}, index=snaps),
            wholesale_links=pd.DataFrame(index=snaps),
            gen_offer_prices=pd.Series({"wind": 0.0, "dear": DEAR_OFFER}),
            gen_bid_prices=pd.Series({"wind": WIND_BID, "dear": DEAR_BID}),
            su_offer_prices=pd.Series({"battery": BATTERY_OFFER}),
            su_bid_prices=pd.Series({"battery": BATTERY_BID}),
            fix_interconnectors=False,
            logger=logging.getLogger("test_bm_dump_load"),
        )
        status, condition = n.optimize(solver_name="highs", extra_functionality=extra)
        assert status == "ok", condition

        d = n.storage_units_t.p_dispatch["battery"]
        s = n.storage_units_t.p_store["battery"]
        # no simultaneous pair - the sink is gone
        assert (d * s).abs().max() < 1e-6
        # and the battery is not used as a sink at all: it is full
        assert s.max() < 1e-6
        # so the surplus is curtailed at the wind farm, 50 MW every hour
        assert n.generators_t.p["wind"].max() <= 100.0 + 1e-6
        assert n.generators_t.p["wind"].min() >= 100.0 - 1e-6

    def test_a_negative_bid_with_the_configured_premium_does_not_churn(self, surplus_network):
        """26 Sep 2026: the storage bid is negative (the battery pays 40 to charge), so a
        simultaneous pair earns the bid back. With the round_trip offer
        (40 / eta_rt + 50 premium) a pair costs (premium + floor) x eta_rt per MWh
        charged for (1 - eta_rt) destroyed, ~217 GBP/MWh here, far above the 90 wind
        bid: the surplus is still curtailed, not burnt."""
        n = surplus_network
        snaps = n.snapshots
        eta_rt = ETA * ETA
        extra = _build_balancing_extra_functionality(
            wholesale_gen=pd.DataFrame({"wind": 150.0, "dear": 0.0}, index=snaps),
            wholesale_su=pd.DataFrame({"battery": 0.0}, index=snaps),
            wholesale_links=pd.DataFrame(index=snaps),
            gen_offer_prices=pd.Series({"wind": 0.0, "dear": DEAR_OFFER}),
            gen_bid_prices=pd.Series({"wind": WIND_BID, "dear": DEAR_BID}),
            su_offer_prices=pd.Series({"battery": 40.0 / eta_rt + 50.0}),
            su_bid_prices=pd.Series({"battery": -40.0}),
            fix_interconnectors=False,
            logger=logging.getLogger("test_bm_dump_load_flipped"),
        )
        status, condition = n.optimize(solver_name="highs", extra_functionality=extra)
        assert status == "ok", condition
        d = n.storage_units_t.p_dispatch["battery"]
        s = n.storage_units_t.p_store["battery"]
        assert (d * s).abs().max() < 1e-6
        assert s.max() < 1e-6 and d.max() < 1e-6
        assert n.generators_t.p["wind"].tolist() == pytest.approx([100.0] * 4, abs=1e-6)

    def test_a_negative_bid_without_a_premium_would_churn(self, surplus_network):
        """Guard on the test above: the same negative bid with a zero premium makes the
        pair nearly free (floor only), and the LP burns energy instead of curtailing,
        which is why market_utils raises the premium (storage_churn_guard)."""
        n = surplus_network
        snaps = n.snapshots
        eta_rt = ETA * ETA
        extra = _build_balancing_extra_functionality(
            wholesale_gen=pd.DataFrame({"wind": 150.0, "dear": 0.0}, index=snaps),
            wholesale_su=pd.DataFrame({"battery": 0.0}, index=snaps),
            wholesale_links=pd.DataFrame(index=snaps),
            gen_offer_prices=pd.Series({"wind": 0.0, "dear": DEAR_OFFER}),
            gen_bid_prices=pd.Series({"wind": WIND_BID, "dear": DEAR_BID}),
            su_offer_prices=pd.Series({"battery": 40.0 / eta_rt}),
            su_bid_prices=pd.Series({"battery": -40.0}),
            fix_interconnectors=False,
            logger=logging.getLogger("test_bm_dump_load_flipped_bare"),
        )
        status, condition = n.optimize(solver_name="highs", extra_functionality=extra)
        assert status == "ok", condition
        d = n.storage_units_t.p_dispatch["battery"]
        s = n.storage_units_t.p_store["battery"]
        assert pd.concat([d, s], axis=1).min(axis=1).sum() > 1.0


@pytest.mark.unit
class TestEndOfWindowStateOfCharge:
    """market.balancing.storage_end_soc: the window must end at or above the Stage 1
    state of charge, softly, with the penalty stripped from the reported cost."""

    FLIPPED_OFFER = 40.0 / (ETA * ETA) + 50.0     # 99.4, below the dear generator
    FLIPPED_BID = -40.0

    def _solve(self, n, penalty):
        gen, su, links = _wholesale_positions(n)
        target = stage1_state_of_charge(n, su).iloc[-1]
        extra = _build_balancing_extra_functionality(
            wholesale_gen=gen,
            wholesale_su=su,
            wholesale_links=links,
            gen_offer_prices=pd.Series({"cheap": CHEAP_OFFER, "dear": DEAR_OFFER}),
            gen_bid_prices=pd.Series({"cheap": CHEAP_BID, "dear": DEAR_BID}),
            su_offer_prices=pd.Series({"battery": self.FLIPPED_OFFER}),
            su_bid_prices=pd.Series({"battery": self.FLIPPED_BID}),
            fix_interconnectors=False,
            logger=logging.getLogger("test_bm_end_soc"),
            storage_end_soc_target=target,
            storage_end_soc_penalty=penalty,
        )
        status, condition = n.optimize(solver_name="highs", extra_functionality=extra)
        assert status == "ok", condition
        return n, target

    def test_stage1_path_is_rebuilt_from_the_net_schedule(self, constrained_network):
        n = constrained_network
        su = pd.DataFrame({"battery": [-10.0, 0.0, 9.0, 0.0]}, index=n.snapshots)
        soc = stage1_state_of_charge(n, su)["battery"]
        assert soc.tolist() == pytest.approx(
            [SOC_INITIAL + 9.0, SOC_INITIAL + 9.0, SOC_INITIAL - 1.0, SOC_INITIAL - 1.0])

    def test_without_the_floor_the_window_empties_storage(self, constrained_network):
        """The battery (offer 99.4) undercuts the dear generator (110) and is drained,
        since nothing values what is left at the end of the window."""
        n, target = self._solve(constrained_network, penalty=0.0)
        assert n.storage_units_t.state_of_charge["battery"].iloc[-1] < target["battery"] - 50.0

    def test_the_floor_keeps_the_stage1_state(self, constrained_network):
        """At 200 GBP/MWh a discharged MWh costs 99.4 + 200 / 0.9 > 110: the dear
        generator runs instead and the battery ends where Stage 1 left it."""
        n, target = self._solve(constrained_network, penalty=200.0)
        end = n.storage_units_t.state_of_charge["battery"].iloc[-1]
        assert end == pytest.approx(target["battery"], abs=1e-6)
        assert _storage_end_soc_shortfall(n, target).sum() == pytest.approx(0.0, abs=1e-6)
        assert n.generators_t.p["dear"].sum() == pytest.approx(4 * 50.0, abs=1e-6)

    def test_a_cheap_floor_is_soft_and_its_penalty_is_stripped(self, constrained_network):
        """At 5 GBP/MWh the slack is used; objective - storage term - penalty x shortfall
        is exactly the redispatch cost, so the penalty never reaches the BM cost."""
        penalty = 5.0
        n, target = self._solve(constrained_network, penalty=penalty)
        shortfall = _storage_end_soc_shortfall(n, target).sum()
        assert shortfall > 1.0
        m = n.model
        inc = m.variables["Generator-increase"].solution.to_pandas()
        dec = m.variables["Generator-decrease"].solution.to_pandas()
        sol = {k: m.variables[f"StorageUnit-{k}"].solution.to_pandas()["battery"].sum()
               for k in ("dispatch-increase", "dispatch-decrease",
                         "store-increase", "store-decrease")}
        redispatch = (
            inc["cheap"].sum() * CHEAP_OFFER + dec["cheap"].sum() * CHEAP_BID
            + inc["dear"].sum() * DEAR_OFFER + dec["dear"].sum() * DEAR_BID
            + (sol["dispatch-increase"] + sol["store-decrease"]) * self.FLIPPED_OFFER
            + (sol["dispatch-decrease"] + sol["store-increase"]) * self.FLIPPED_BID
        )
        slack = float(m.variables["StorageUnit-end_soc_slack"].solution.sum())
        assert slack == pytest.approx(shortfall, abs=1e-6)
        assert n.objective - _storage_energy_cost(n) - penalty * shortfall == pytest.approx(
            redispatch, rel=1e-6)

    def test_an_unreachable_target_is_not_infeasible(self, constrained_network):
        """A target above capacity is met by slack, never by infeasibility."""
        n = constrained_network
        gen, su, links = _wholesale_positions(n)
        extra = _build_balancing_extra_functionality(
            wholesale_gen=gen, wholesale_su=su, wholesale_links=links,
            gen_offer_prices=pd.Series({"cheap": CHEAP_OFFER, "dear": DEAR_OFFER}),
            gen_bid_prices=pd.Series({"cheap": CHEAP_BID, "dear": DEAR_BID}),
            su_offer_prices=pd.Series({"battery": self.FLIPPED_OFFER}),
            su_bid_prices=pd.Series({"battery": self.FLIPPED_BID}),
            fix_interconnectors=False, logger=logging.getLogger("test_bm_end_soc"),
            storage_end_soc_target=pd.Series({"battery": 10_000.0}),
            storage_end_soc_penalty=200.0,
        )
        status, condition = n.optimize(solver_name="highs", extra_functionality=extra)
        assert status == "ok", condition


@pytest.mark.unit
class TestReportedCostExcludesTheTerm:

    def test_objective_minus_storage_term_is_pure_redispatch(self, constrained_network):
        """n.objective = redispatch cost + sum(mc * p_dispatch * h); the helper
        recovers the second part so the reported BM cost is the first."""
        n = _solve_bm(constrained_network)
        m = n.model
        inc = m.variables["Generator-increase"].solution.to_pandas()
        dec = m.variables["Generator-decrease"].solution.to_pandas()
        sol = {k: m.variables[f"StorageUnit-{k}"].solution.to_pandas()["battery"].sum()
               for k in ("dispatch-increase", "dispatch-decrease",
                         "store-increase", "store-decrease")}
        # net output up (more discharge / less charging) is an offer,
        # net output down (less discharge / more charging) is a bid
        redispatch = (
            inc["cheap"].sum() * CHEAP_OFFER + dec["cheap"].sum() * CHEAP_BID
            + inc["dear"].sum() * DEAR_OFFER + dec["dear"].sum() * DEAR_BID
            + (sol["dispatch-increase"] + sol["store-decrease"]) * BATTERY_OFFER
            + (sol["dispatch-decrease"] + sol["store-increase"]) * BATTERY_BID
        )
        storage_term = _storage_energy_cost(n)
        assert storage_term == pytest.approx(
            STORAGE_MC * n.storage_units_t.p_dispatch["battery"].sum(), rel=1e-9)
        assert storage_term > 0.0
        assert n.objective - storage_term == pytest.approx(redispatch, rel=1e-6)

    def test_helper_is_zero_without_storage(self):
        n = pypsa.Network()
        n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
        n.add("Bus", "A")
        assert _storage_energy_cost(n) == 0.0


@pytest.mark.unit
class TestReverseLegGuard:
    """In an hour Stage 1 charges, discharging more costs the same offer as charging
    less but burns (1/eta_d - eta_s) per MWh; the objective-only reverse-leg price
    (churn guard x that loss) breaks the tie, and is stripped from the reported cost."""

    def test_price_is_the_guard_times_the_burnt_energy(self, constrained_network):
        from scripts.market.solve_balancing import storage_reverse_leg_prices

        prices = storage_reverse_leg_prices(
            constrained_network, {"storage_reverse_leg_guard_gbp_per_mwh": 250.0})
        assert prices["battery"] == pytest.approx(250.0 * (1 / ETA - ETA))
        assert storage_reverse_leg_prices(
            constrained_network, {"storage_reverse_leg_guard_gbp_per_mwh": 0.0}).empty

    def test_a_genuine_flip_pays_the_guard_only_in_the_objective(self, constrained_network):
        """Stage 1 charges the battery at B at 10 MW; the BM needs 50 MW more at B, so it
        stops charging (charge less, offer) and, with the peaker at 300, discharges (the reverse leg). No
        unit-hour both charges and discharges, and objective - storage term - reverse-leg
        cost is the redispatch cost priced at the ordinary bids and offers."""
        from scripts.market.solve_balancing import (
            _storage_reverse_leg_cost,
            storage_reverse_leg_prices,
        )

        n = constrained_network
        snaps = n.snapshots
        gen = pd.DataFrame({"cheap": 160.0, "dear": 0.0}, index=snaps)
        su = pd.DataFrame({"battery": -10.0}, index=snaps)
        prices = storage_reverse_leg_prices(n, {"storage_reverse_leg_guard_gbp_per_mwh": 250.0})
        offer, bid = 40.0 / (ETA * ETA) + 67.0, 44.0
        # dear enough that discharging (offer + reverse-leg price ~169) still wins
        dear_offer = 300.0
        extra = _build_balancing_extra_functionality(
            wholesale_gen=gen, wholesale_su=su, wholesale_links=pd.DataFrame(index=snaps),
            gen_offer_prices=pd.Series({"cheap": CHEAP_OFFER, "dear": dear_offer}),
            gen_bid_prices=pd.Series({"cheap": CHEAP_BID, "dear": DEAR_BID}),
            su_offer_prices=pd.Series({"battery": offer}),
            su_bid_prices=pd.Series({"battery": bid}),
            fix_interconnectors=False, logger=logging.getLogger("test_bm_reverse_leg"),
            storage_reverse_leg_price=prices,
        )
        status, condition = n.optimize(solver_name="highs", extra_functionality=extra)
        assert status == "ok", condition
        d = n.storage_units_t.p_dispatch["battery"]
        s = n.storage_units_t.p_store["battery"]
        assert (d * s).abs().max() < 1e-6
        assert d.sum() > 1.0                      # the flip happened
        m = n.model
        inc = m.variables["Generator-increase"].solution.to_pandas()
        dec = m.variables["Generator-decrease"].solution.to_pandas()
        sol = {k: m.variables[f"StorageUnit-{k}"].solution.to_pandas()["battery"].sum()
               for k in ("dispatch-increase", "dispatch-decrease",
                         "store-increase", "store-decrease")}
        redispatch = (
            inc["cheap"].sum() * CHEAP_OFFER + dec["cheap"].sum() * CHEAP_BID
            + inc["dear"].sum() * dear_offer + dec["dear"].sum() * DEAR_BID
            + (sol["dispatch-increase"] + sol["store-decrease"]) * offer
            + (sol["dispatch-decrease"] + sol["store-increase"]) * bid
        )
        reverse = _storage_reverse_leg_cost(n, su, prices)
        assert reverse == pytest.approx(prices["battery"] * d.sum(), rel=1e-9)
        assert n.objective - _storage_energy_cost(n) - reverse == pytest.approx(redispatch, rel=1e-6)

    def test_code_default_matches_defaults_yaml(self):
        from pathlib import Path

        import yaml

        from scripts.market.solve_balancing import (
            DEFAULT_STORAGE_END_SOC_PENALTY,
            DEFAULT_STORAGE_REVERSE_LEG_GUARD,
        )
        from scripts.market.market_utils import DEFAULT_STORAGE_CHURN_GUARD

        root = Path(__file__).resolve().parents[2]
        balancing = load_defaults()["market"]["balancing"]
        assert balancing["storage_reverse_leg_guard_gbp_per_mwh"] == DEFAULT_STORAGE_REVERSE_LEG_GUARD
        assert balancing["storage_churn_guard_gbp_per_mwh"] == DEFAULT_STORAGE_CHURN_GUARD
        assert balancing["storage_end_soc"]["penalty_gbp_per_mwh"] == DEFAULT_STORAGE_END_SOC_PENALTY
