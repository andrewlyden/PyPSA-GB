"""Item OW3 (3 Oct 2026): hydrogen-to-power fuel at CP30's hydrogen price, and the turbine VOM.

* ``hydrogen.turbine_fuel.source: external_supply`` gives the turbines their own
  bus, fed by a full tank (a Store) priced per MWh drawn, so turbine fuel is
  never electrolysed and never serves the offtake.
* ``hydrogen.turbine_variable_om.basis: electricity_output`` puts the VOM on the
  electricity leg (VOM x efficiency on p0, hydrogen in).
* The code defaults reproduce the build before the item; the brownfield overlay switches
  both on, with the turbine fuel price equal to the store's end-of-window value.
* Stage 1's hydrogen value (the BM's electrolyser price) ignores the fuel bus.
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import pypsa
import pytest
import yaml

from scripts.hydrogen.add_hydrogen_system import (
    H2_POWER_FUEL_BUS,
    H2_POWER_FUEL_CARRIER,
    H2_POWER_FUEL_STORE,
    H2_POWER_FUEL_TANK_HOURS,
    H2_TURBINE_EFFICIENCY,
    H2_TURBINE_MARGINAL_COST,
    add_hydrogen_system,
    resolve_turbine_settings,
    turbine_link_marginal_cost,
)
from scripts.market.market_utils import (
    DEFAULT_HYDROGEN_BUS_CARRIERS,
    hydrogen_links,
    wholesale_hydrogen_price,
)
from config.config_loader import load_defaults

ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger("test_cp30_hydrogen_turbine_fuel")
CP30_MONTH = "CP30_FFR_2030_Jan_cp30_network"
# CP30's hydrogen price (1.2 x CP.22 gas, gross CV) in the LHV units of the GB_H2 system.
CP30_PRICE_LHV = round(41.2 * 141.88 / 119.96, 2)
EXTERNAL = {
    "turbine_fuel": {"source": "external_supply", "price_gbp_per_mwh_h2": CP30_PRICE_LHV},
    "turbine_variable_om": {"gbp_per_mwh": 5.0, "basis": "electricity_output"},
}
OFFTAKE_MW_H2 = 30.0


def _network(gas_price: float, hours: int = 3) -> pypsa.Network:
    """One AC bus, gas at ``gas_price``, a 100 MW load and a 50 MW hydrogen generator."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=hours, freq="h"))
    n.add("Bus", "E", x=0.0, y=0.0, carrier="AC")
    n.add("Generator", "gas", bus="E", carrier="CCGT", p_nom=1000.0, marginal_cost=gas_price)
    n.add("Load", "demand", bus="E", p_set=100.0)
    n.add("Generator", "h2", bus="E", carrier="H2", p_nom=50.0)
    return n


def _build(n, settings=None):
    return add_hydrogen_system(
        n, electrolysis_capacity_mw=200.0, h2_generation_capacity_mw=50.0, logger=LOGGER,
        exogenous_h2_demand_mwh=OFFTAKE_MW_H2 * 8760.0, turbine_settings=settings,
    )


@pytest.mark.unit
class TestSettings:

    def test_empty_config_is_the_build_before_ow3(self):
        s = resolve_turbine_settings({})
        assert s == {"fuel_source": "electrolysis", "fuel_price_gbp_per_mwh_h2": None,
                     "vom_basis": "hydrogen_input", "vom_gbp_per_mwh": H2_TURBINE_MARGINAL_COST,
                     "availability": 1.0}
        assert resolve_turbine_settings(None) == s

    def test_external_supply_needs_a_price(self):
        with pytest.raises(ValueError, match="price_gbp_per_mwh_h2"):
            resolve_turbine_settings({"turbine_fuel": {"source": "external_supply"}})
        with pytest.raises(ValueError, match="source"):
            resolve_turbine_settings({"turbine_fuel": {"source": "imports"}})
        with pytest.raises(ValueError, match="basis"):
            resolve_turbine_settings({"turbine_variable_om": {"basis": "per_mwh"}})

    def test_vom_on_the_electricity_leg_is_vom_times_efficiency_on_p0(self):
        assert turbine_link_marginal_cost(5.0, "electricity_output", 0.5) == pytest.approx(2.5)
        assert turbine_link_marginal_cost(5.0, "hydrogen_input", 0.5) == pytest.approx(5.0)


@pytest.mark.unit
class TestBuild:

    def test_default_build_is_unchanged(self):
        n = _build(_network(120.0))
        turbine = n.links.loc["H2_turbine_h2"]
        assert turbine["bus0"] == "GB_H2"
        assert turbine["marginal_cost"] == pytest.approx(H2_TURBINE_MARGINAL_COST)
        assert H2_POWER_FUEL_BUS not in n.buses.index
        assert list(n.stores.index) == ["GB_H2_storage"]

    def test_external_supply_topology(self):
        n = _build(_network(120.0), resolve_turbine_settings(EXTERNAL))
        turbine = n.links.loc["H2_turbine_h2"]
        assert turbine["bus0"] == H2_POWER_FUEL_BUS and turbine["bus1"] == "E"
        assert turbine["marginal_cost"] == pytest.approx(5.0 * H2_TURBINE_EFFICIENCY)
        # The fuel bus is hydrogen (so the BM holds the turbines), fed by nothing but the tank.
        assert n.buses.at[H2_POWER_FUEL_BUS, "carrier"] == "H2_gas"
        assert not (n.links["bus1"] == H2_POWER_FUEL_BUS).any()
        tank = n.stores.loc[H2_POWER_FUEL_STORE]
        assert tank["bus"] == H2_POWER_FUEL_BUS and tank["carrier"] == H2_POWER_FUEL_CARRIER
        assert tank["marginal_cost"] == pytest.approx(CP30_PRICE_LHV)
        assert tank["e_nom"] == pytest.approx(50.0 / H2_TURBINE_EFFICIENCY * H2_POWER_FUEL_TANK_HOURS)
        assert tank["e_initial"] == pytest.approx(tank["e_nom"]) and not tank["e_cyclic"]
        # The electrolysis store, the offtake and the electrolysers are as before.
        assert n.stores.at["GB_H2_storage", "bus"] == "GB_H2"
        assert n.stores.at["GB_H2_storage", "e_nom"] == pytest.approx(50.0 / 0.5 * 168)
        assert n.loads.at["GB_H2_exogenous_demand", "p_set"] == pytest.approx(OFFTAKE_MW_H2)
        assert set(hydrogen_links(n, ("electrolysis", "H2_turbine"), DEFAULT_HYDROGEN_BUS_CARRIERS)) == {
            "electrolysis_E", "H2_turbine_h2"}


def _solve(n):
    status, _ = n.optimize(solver_name="highs")
    assert status == "ok"
    return n


@pytest.mark.unit
class TestDispatch:
    """Gas sets the price; hydrogen made from it costs gas / 0.7 per MWh_H2."""

    def test_electrolysed_fuel_keeps_turbines_off_at_a_gas_price(self):
        n = _solve(_build(_network(120.0)))
        # Fuel from gas costs 120 / 0.7 / 0.5 + 5 / 0.5 = 352.9 per MWh_el: never in merit.
        assert n.links_t.p0["H2_turbine_h2"].sum() == pytest.approx(0.0, abs=1e-6)
        assert n.links_t.p0["electrolysis_E"].sum() * 0.7 == pytest.approx(3 * OFFTAKE_MW_H2, rel=1e-3)

    def test_bought_fuel_runs_turbines_above_the_hurdle_and_leaves_electrolysis_to_the_offtake(self):
        n = _solve(_build(_network(120.0), resolve_turbine_settings(EXTERNAL)))
        hurdle = (CP30_PRICE_LHV + 2.5) / H2_TURBINE_EFFICIENCY
        assert hurdle == pytest.approx(102.46, abs=0.01) and hurdle < 120.0
        burn = n.links_t.p0["H2_turbine_h2"]
        assert burn.tolist() == pytest.approx([100.0] * 3)                       # 50 MW_el at 0.5
        # The tank supplies exactly the burn; electrolysis serves the offtake alone.
        assert n.stores_t.p[H2_POWER_FUEL_STORE].tolist() == pytest.approx(burn.tolist())
        assert n.links_t.p0["electrolysis_E"].sum() * 0.7 == pytest.approx(3 * OFFTAKE_MW_H2, rel=1e-3)
        # Objective: gas for load less turbine output plus electrolysis, the bought fuel and VOM.
        electrolysis = n.links_t.p0["electrolysis_E"].sum()
        expected = 120.0 * (300.0 - 150.0 + electrolysis) + (CP30_PRICE_LHV + 2.5) * 300.0
        assert n.objective == pytest.approx(expected, rel=1e-6)

    def test_below_the_hurdle_bought_fuel_stays_in_the_tank(self):
        n = _solve(_build(_network(100.0), resolve_turbine_settings(EXTERNAL)))
        assert n.links_t.p0["H2_turbine_h2"].sum() == pytest.approx(0.0, abs=1e-6)
        assert n.stores_t.p[H2_POWER_FUEL_STORE].abs().sum() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_stage1_hydrogen_value_ignores_the_fuel_bus():
    n = _build(_network(120.0), resolve_turbine_settings(EXTERNAL))
    prices = pd.DataFrame({"E": [120.0] * 3, "GB_H2": [171.4] * 3, H2_POWER_FUEL_BUS: [CP30_PRICE_LHV] * 3},
                          index=n.snapshots)
    assert wholesale_hydrogen_price(prices, n).tolist() == pytest.approx([171.4] * 3)
    # Without electrolysers every hydrogen bus counts, as before.
    n.remove("Link", ["electrolysis_E"])
    expected = (171.4 + CP30_PRICE_LHV) / 2.0
    assert wholesale_hydrogen_price(prices, n).tolist() == pytest.approx([expected] * 3)


@pytest.mark.unit
class TestMarketPipeline:
    """The market stages see the fuel bus as one more hydrogen island."""

    def test_copperplate_keeps_the_fuel_bus_and_the_turbine(self):
        from scripts.market.copperplate import collapse_mainland

        n = _build(_network(120.0), resolve_turbine_settings(EXTERNAL))
        n.add("Bus", "E2", x=1.0, y=1.0, carrier="AC")
        n.add("Line", "E_E2", bus0="E", bus1="E2", x=0.1, r=0.01, s_nom=1e6)
        collapsed, _ = collapse_mainland(n, LOGGER)
        assert H2_POWER_FUEL_BUS in collapsed.buses.index
        assert collapsed.links.at["H2_turbine_h2", "bus0"] == H2_POWER_FUEL_BUS
        assert collapsed.stores.at[H2_POWER_FUEL_STORE, "bus"] == H2_POWER_FUEL_BUS

    def test_bm_store_path_ignores_turbines_on_bought_fuel(self):
        from scripts.market.solve_balancing import stage1_store_levels

        n = _build(_network(120.0, hours=2), resolve_turbine_settings(EXTERNAL))
        links = pd.DataFrame({"electrolysis_E": [100.0, 0.0], "H2_turbine_h2": [100.0, 100.0]}, index=n.snapshots)
        start = pd.Series({"GB_H2_storage": 500.0})
        n.stores.loc["GB_H2_storage", "standing_loss"] = 0.0
        path = stage1_store_levels(n, links, start)
        # +70 - 30 offtake, then -30: the turbines' 200 MWh of fuel never touches GB_H2.
        assert path["GB_H2_storage"].tolist() == pytest.approx([540.0, 510.0])

    def test_held_turbines_burn_bought_fuel_in_the_rolling_bm(self):
        from scripts.market.solve_balancing import solve_rolling_balancing, stage1_store_band

        n = _build(_network(120.0, hours=4), resolve_turbine_settings(EXTERNAL))
        n.stores.loc["GB_H2_storage", ["e_initial", "e_cyclic", "standing_loss"]] = [500.0, False, 0.0]
        snaps = n.snapshots
        gen = pd.DataFrame({"gas": 100.0 - 50.0 + 60.0}, index=snaps)
        links = pd.DataFrame({"electrolysis_E": 60.0, "H2_turbine_h2": 100.0}, index=snaps)
        offer, bid = pd.Series({"gas": 130.0}), pd.Series({"gas": -110.0})
        start = pd.Series({"GB_H2_storage": 500.0})
        band = stage1_store_band(n, links, start, 1e-3)
        result = solve_rolling_balancing(
            n, gen, pd.DataFrame(index=snaps), links, offer, bid, pd.Series(dtype=float), pd.Series(dtype=float),
            {"window_hours": 2, "fix_interconnectors": True}, "highs", {}, {}, LOGGER,
            anchored_links=[], fixed_links=["H2_turbine_h2"], store_start=start, store_band=band,
            store_penalty=200.0,
        )
        assert result is not None
        assert n.links_t.p0["H2_turbine_h2"].tolist() == pytest.approx([100.0] * 4, abs=1e-4)
        # The rolling BM records carried stores' levels only (solve_balancing.py, the
        # stitched stores_t.e); the tank is not carried, so its column is empty.
        assert n.stores_t.e[H2_POWER_FUEL_STORE].isna().all()
        # The electrolysis store moves with the electrolysers and the offtake only.
        e = n.stores_t.e["GB_H2_storage"]
        net = 0.7 * n.links_t.p0["electrolysis_E"] - OFFTAKE_MW_H2
        assert e.tolist() == pytest.approx((500.0 + net.cumsum()).tolist(), abs=1e-4)


@pytest.mark.unit
class TestConfig:

    @staticmethod
    def _scenarios():
        sys.path.insert(0, str(ROOT / "config"))
        from config_loader import load_config

        return load_config()["scenarios"]

    def test_global_defaults_keep_the_build_before_ow3(self):
        defaults = load_defaults()
        assert "turbine_fuel" not in (defaults.get("hydrogen") or {})
        assert "turbine_variable_om" not in (defaults.get("hydrogen") or {})
        assert defaults["market"]["hydrogen_store"]["end_of_window_value_gbp_per_mwh_h2"] is None

    def test_cp30_months_buy_turbine_fuel_at_the_store_value(self):
        scenario = self._scenarios()[CP30_MONTH]
        settings = resolve_turbine_settings(scenario["hydrogen"])
        assert settings["fuel_source"] == "external_supply"
        assert settings["vom_basis"] == "electricity_output"
        assert settings["vom_gbp_per_mwh"] == pytest.approx(H2_TURBINE_MARGINAL_COST)
        value = scenario["market"]["hydrogen_store"]["end_of_window_value_gbp_per_mwh_h2"]
        assert settings["fuel_price_gbp_per_mwh_h2"] == pytest.approx(value) == pytest.approx(CP30_PRICE_LHV)
        # The BM keeps holding the turbines, which the fuel bus's H2_gas carrier relies on.
        assert "H2_turbine" in scenario["market"]["balancing"]["hydrogen_links"]["hold_carriers"]
        assert "H2_gas" in scenario["market"]["balancing"]["hydrogen_links"]["bus_carriers"]

    def test_only_cp30_scenarios_change(self):
        for sid, scenario in self._scenarios().items():
            hydrogen = scenario.get("hydrogen") or {}
            if (hydrogen.get("fes") or {}).get("enabled") or str(scenario.get("scenario_source")) != "CP30":
                assert resolve_turbine_settings(hydrogen)["fuel_source"] == "electrolysis", sid
