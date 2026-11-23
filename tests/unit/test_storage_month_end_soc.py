"""Each monthly market run hands back the storage energy it was given.

Each CP30 month starts every storage unit at half its energy capacity and the
rolling day-ahead stage has no terminal value, so day 1 sold the 74 GWh it was
handed: 0.89 TWh a year of free energy (audit of 26 Sep 2026). The month-end
condition holds the final state of charge at or above the start in the window
that holds the last snapshot; these tests pin that, the feasibility cap, the
starting-level override, and the off switch.
"""

import logging

import pandas as pd
import pypsa
import pytest

from scripts.market.market_utils import (
    apply_initial_storage_soc,
    build_month_end_soc_callback,
    resolve_storage_soc_config,
)

LOGGER = logging.getLogger("test_month_end_soc")


@pytest.fixture
def storage_network():
    """Load 50 MW served by gas (mc 80); a 20 MW / 4 h battery starts with 40 MWh."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-04-30", periods=6, freq="h"))
    n.add("Bus", "bus", carrier="AC")
    n.add("Generator", "gas", bus="bus", p_nom=200.0, marginal_cost=80.0, carrier="CCGT")
    n.add("Load", "load", bus="bus", p_set=50.0)
    n.add("StorageUnit", "battery", bus="bus", p_nom=20.0, max_hours=4.0, carrier="Battery",
          efficiency_store=0.95, efficiency_dispatch=0.95, state_of_charge_initial=40.0,
          marginal_cost=1.0, cyclic_state_of_charge=False)
    return n


def _final_soc(n, market_config, final=None):
    final = n.snapshots[-1] if final is None else final
    callback = build_month_end_soc_callback(n, market_config, final, LOGGER)
    status, condition = n.optimize(solver_name="highs", extra_functionality=callback)
    assert status == "ok", condition
    return float(n.storage_units_t.state_of_charge["battery"].iloc[-1])


@pytest.mark.unit
class TestMonthEndStateOfCharge:

    def test_free_month_end_sells_the_starting_energy(self, storage_network):
        """The defect: nothing asks for the 40 MWh back, so it is sold against gas."""
        assert _final_soc(storage_network, {"storage_soc": {"month_end": "free"}}) == pytest.approx(0.0, abs=1e-6)

    def test_month_end_holds_the_start(self, storage_network):
        assert _final_soc(storage_network, {}) == pytest.approx(40.0, abs=1e-6)

    def test_inert_outside_the_last_window(self, storage_network):
        later = storage_network.snapshots[-1] + pd.Timedelta(hours=1)
        assert _final_soc(storage_network, {}, final=later) == pytest.approx(0.0, abs=1e-6)

    def test_target_is_capped_at_what_the_window_can_recharge(self, storage_network):
        # 1 MW of charging over 6 h cannot refill 40 MWh; the target becomes
        # 0.95 x 1 x 0.95 x 6 = 5.4 MWh rather than an infeasible 40.
        storage_network.storage_units.loc["battery", "p_min_pu"] = -0.05
        final = _final_soc(storage_network, {})
        assert final == pytest.approx(0.95 * 1.0 * 0.95 * 6.0, rel=1e-6)

    def test_initial_fraction_sets_the_start_for_both_stages(self, storage_network):
        initial = apply_initial_storage_soc(storage_network, {"storage_soc": {"initial_fraction": 0.25}}, LOGGER)
        assert initial["battery"] == pytest.approx(20.0)
        assert storage_network.storage_units.at["battery", "state_of_charge_initial"] == pytest.approx(20.0)
        assert _final_soc(storage_network, {}) == pytest.approx(20.0, abs=1e-6)

    def test_null_fraction_keeps_the_network_value(self, storage_network):
        initial = apply_initial_storage_soc(storage_network, {}, LOGGER)
        assert initial["battery"] == pytest.approx(40.0)

    def test_config_validation(self):
        assert resolve_storage_soc_config({})["month_end"] == "at_least_initial"
        with pytest.raises(ValueError):
            resolve_storage_soc_config({"storage_soc": {"month_end": "sometimes"}})
        with pytest.raises(ValueError):
            resolve_storage_soc_config({"storage_soc": {"initial_fraction": 1.5}})
