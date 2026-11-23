"""ES1 output calibration of the CP30 FFR 2030 fleet (gas_cal, 14 Oct 2026).

Stage 1 s1_rb ran the baseload fleet and the hydrogen turbines above CP30 ES1 and
gas CHP below it. The calibration moves output only, never capacity:

* ``hydrogen.turbine_availability`` is the turbine links' p_max_pu (p_nom stays
  CP30's 276.1 MW of electricity);
* ``thermal.availability_factors.group_overrides`` sets a cap or floor per
  (carrier, data_source), so transmission biomass (the station schedule) and the
  CP30 proxy can follow their own ES1 lines;
* the FFR 2030 scenario carries the calibrated caps and floors (nuclear, waste,
  CCS Biomass, CHP floor, biomass groups, hydrogen availability).
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import pypsa
import pytest

from scripts.generators.apply_marginal_costs import seasonal_heat_factors
from scripts.generators.integrate_thermal_generators import (
    apply_thermal_group_overrides,
    check_min_output_within_availability,
)
from scripts.hydrogen.add_hydrogen_system import (
    H2_TURBINE_EFFICIENCY,
    add_hydrogen_system,
    resolve_turbine_settings,
)

ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger("test_cp30_es1_output_calibration")
SHARED = "CP30_FFR_2030"
MONTH = "CP30_FFR_2030_Jan_cp30_network"
ND_SHARED = "CP30_ND_2030_ETYS_market_neso_es5_landing_reassignment_reinforced"
CP30_ES1 = ROOT / "resources" / "CP30" / "CP30_ES1.csv"
PATHWAY = "Further Flex and Renewables"
EXTERNAL = {
    "turbine_fuel": {"source": "external_supply", "price_gbp_per_mwh_h2": 48.73},
    "turbine_variable_om": {"gbp_per_mwh": 5.0, "basis": "electricity_output"},
}


# ── hydrogen turbine availability ───────────────────────────────────────────

def _h2_network(hours: int = 3) -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=hours, freq="h"))
    n.add("Bus", "E", x=0.0, y=0.0, carrier="AC")
    n.add("Generator", "gas", bus="E", carrier="CCGT", p_nom=1000.0, marginal_cost=120.0)
    n.add("Load", "demand", bus="E", p_set=100.0)
    n.add("Generator", "h2", bus="E", carrier="H2", p_nom=50.0)
    return n


def _build(settings):
    return add_hydrogen_system(_h2_network(), electrolysis_capacity_mw=200.0, h2_generation_capacity_mw=50.0,
                               logger=LOGGER, exogenous_h2_demand_mwh=30.0 * 8760.0, turbine_settings=settings)


@pytest.mark.unit
class TestTurbineAvailability:

    def test_default_is_no_derating(self):
        assert resolve_turbine_settings({})["availability"] == 1.0
        n = _build(resolve_turbine_settings(EXTERNAL))
        assert n.links.at["H2_turbine_h2", "p_max_pu"] == pytest.approx(1.0)

    def test_out_of_range_is_refused(self):
        for bad in (0.0, -0.1, 1.2):
            with pytest.raises(ValueError, match="turbine_availability"):
                resolve_turbine_settings({**EXTERNAL, "turbine_availability": bad})

    def test_availability_moves_output_not_capacity(self):
        settings = resolve_turbine_settings({**EXTERNAL, "turbine_availability": 0.642})
        n = _build(settings)
        link = n.links.loc["H2_turbine_h2"]
        assert link["p_nom"] * link["efficiency"] == pytest.approx(50.0)      # capacity unchanged
        assert link["p_max_pu"] == pytest.approx(0.642)
        status, _ = n.optimize(solver_name="highs")
        assert status == "ok"
        # Above the 102.46 hurdle the turbines run at their available output only.
        out = n.links_t.p0["H2_turbine_h2"] * H2_TURBINE_EFFICIENCY
        assert out.tolist() == pytest.approx([50.0 * 0.642] * 3, rel=1e-6)


# ── group overrides ─────────────────────────────────────────────────────────

def _biomass_network() -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    n.add("Bus", "b")
    n.add("Generator", "drax", bus="b", carrier="biomass", p_nom=1995.0, p_max_pu=0.80, p_min_pu=0.35,
          data_source="CP30_biomass_station_schedule")
    n.add("Generator", "proxy_1", bus="b", carrier="biomass", p_nom=20.0, p_max_pu=0.80, p_min_pu=0.35,
          data_source="CP30")
    n.add("Generator", "proxy_2", bus="b", carrier="biomass", p_nom=30.0, p_max_pu=0.80, p_min_pu=0.35,
          data_source="CP30")
    n.add("Generator", "efw", bus="b", carrier="waste_to_energy", p_nom=50.0, p_max_pu=0.62, p_min_pu=0.55,
          data_source="CP30")
    return n


@pytest.mark.unit
class TestGroupOverrides:

    def test_sets_cap_and_floor_by_data_source_only(self):
        n = _biomass_network()
        report = apply_thermal_group_overrides(n, [
            {"carrier": "biomass", "data_source": "CP30_biomass_station_schedule", "min_output": 0.396},
            {"carrier": "biomass", "data_source": "CP30", "factor": 0.382},
        ], LOGGER)
        g = n.generators
        assert g.at["drax", "p_min_pu"] == pytest.approx(0.396) and g.at["drax", "p_max_pu"] == pytest.approx(0.80)
        assert g.loc[["proxy_1", "proxy_2"], "p_max_pu"].tolist() == pytest.approx([0.382, 0.382])
        assert g.loc[["proxy_1", "proxy_2"], "p_min_pu"].tolist() == pytest.approx([0.35, 0.35])
        # Another carrier with the same data_source is untouched; capacity never moves.
        assert g.at["efw", "p_max_pu"] == pytest.approx(0.62)
        assert g["p_nom"].tolist() == pytest.approx([1995.0, 20.0, 30.0, 50.0])
        assert report["generators"].tolist() == [1, 2]
        assert report["p_nom_mw"].tolist() == pytest.approx([1995.0, 50.0])

    def test_empty_is_a_no_op(self):
        n = _biomass_network()
        before = n.generators.copy()
        assert apply_thermal_group_overrides(n, [], LOGGER).empty
        pd.testing.assert_frame_equal(n.generators, before)

    def test_refusals(self):
        with pytest.raises(ValueError, match="matches no generator"):
            apply_thermal_group_overrides(_biomass_network(), [
                {"carrier": "biomass", "data_source": "DUKES", "factor": 0.5}], LOGGER)
        with pytest.raises(ValueError, match="above the availability factor"):
            apply_thermal_group_overrides(_biomass_network(), [
                {"carrier": "biomass", "data_source": "CP30", "factor": 0.30}], LOGGER)
        with pytest.raises(ValueError, match="unknown keys"):
            apply_thermal_group_overrides(_biomass_network(), [
                {"carrier": "biomass", "data_source": "CP30", "p_nom": 10.0}], LOGGER)
        with pytest.raises(ValueError, match="neither"):
            apply_thermal_group_overrides(_biomass_network(), [
                {"carrier": "biomass", "data_source": "CP30"}], LOGGER)


# ── the FFR 2030 scenario ───────────────────────────────────────────────────

def _scenarios():
    sys.path.insert(0, str(ROOT / "config"))
    from config_loader import load_config

    return load_config(include_archive=True)["scenarios"]   # CP30 New Dispatch is archived


@pytest.mark.unit
class TestScenarioCalibration:

    def test_shared_network_carries_the_es1_calibration(self):
        scenario = _scenarios()[SHARED]
        avail = scenario["thermal"]["availability_factors"]
        check_min_output_within_availability(avail)
        assert avail["factors"]["nuclear"] == pytest.approx(0.854)
        assert avail["factors"]["waste_to_energy"] == pytest.approx(0.607)
        assert avail["factors"]["CCS Biomass"] == pytest.approx(0.958)
        assert avail["min_output"]["CCS Biomass"] == pytest.approx(avail["factors"]["CCS Biomass"])
        assert avail["min_output"]["CHP"] == pytest.approx(0.345)
        # The heat-led CHP floor at its winter peak stays under the 0.45 cap.
        seasonal = scenario["marginal_costs"]["byproduct_adjustments"]["CHP"]["seasonal"]
        peak = seasonal_heat_factors(pd.date_range("2030-01-01", "2030-12-31 23:00", freq="h"), seasonal).max()
        assert avail["min_output"]["CHP"] * peak <= avail["factors"]["CHP"]
        groups = {(e["carrier"], e["data_source"]): e for e in avail["group_overrides"]}
        assert groups[("biomass", "CP30_biomass_station_schedule")]["min_output"] == pytest.approx(0.396)
        proxy = groups[("biomass", "CP30")]["factor"]
        assert proxy == pytest.approx(0.382) and proxy >= avail["min_output"]["biomass"]
        assert resolve_turbine_settings(scenario["hydrogen"])["availability"] == pytest.approx(0.642)
        # The turbine fuel settings are still the brownfield overlay's (deep merge).
        assert resolve_turbine_settings(scenario["hydrogen"])["fuel_source"] == "external_supply"

    def test_energy_windows_are_kept(self):
        avail = _scenarios()[SHARED]["thermal"]["availability_factors"]
        assert avail["min_output_energy_window_hours"] == {
            "biomass": 24, "waste_to_energy": 24, "CHP": 24, "biogas": 24}

    def test_new_dispatch_keeps_the_package_before_the_calibration(self):
        scenario = _scenarios()[ND_SHARED]
        avail = scenario["thermal"]["availability_factors"]
        assert "group_overrides" not in avail and "CCS Biomass" not in avail["factors"]
        assert avail["factors"]["nuclear"] == pytest.approx(0.90)
        assert avail["min_output"]["CHP"] == pytest.approx(0.30)
        assert avail["min_output"]["CCS Biomass"] == pytest.approx(1.0)
        assert "turbine_availability" not in scenario["hydrogen"]
        assert resolve_turbine_settings(scenario["hydrogen"])["availability"] == 1.0

    def test_monthly_cases_solve_on_the_same_calibration(self):
        scenarios = _scenarios()
        shared, month = scenarios[SHARED], scenarios[MONTH]
        assert month["thermal"]["availability_factors"] == shared["thermal"]["availability_factors"]
        assert month["hydrogen"] == shared["hydrogen"]


@pytest.mark.skipif(not CP30_ES1.exists(), reason="CP30 ES1 extract not built")
@pytest.mark.unit
def test_es1_targets_behind_the_values():
    es1 = pd.read_csv(CP30_ES1)
    rows = es1[es1["Pathway"].eq(PATHWAY)]

    def total(variable, subtypes):
        sel = rows[rows["Variable"].eq(variable) & rows["SubType"].isin(subtypes)]
        return float(pd.to_numeric(sel["2030"], errors="coerce").sum())

    assert total("Capacity (MW)", ["Hydrogen", "Hydrogen CHP"]) == pytest.approx(276.10, abs=0.01)
    assert total("Generation (GWh)", ["Hydrogen", "Hydrogen CHP"]) == pytest.approx(338.40, abs=0.01)
    assert total("Generation (GWh)", ["Biomass", "Biomass CHP"]) == pytest.approx(12155.08, abs=0.01)
    assert total("Generation (GWh)", ["CCS Biomass"]) / (450.0 * 8.76) == pytest.approx(0.8739, abs=1e-4)
    assert total("Generation (GWh)", ["Waste", "Waste CHP"]) == pytest.approx(12824.03, abs=0.01)
    assert total("Generation (GWh)", ["Gas CHP"]) == pytest.approx(11782.46, abs=0.01)
    assert total("Generation (GWh)", ["Nuclear - Large"]) == pytest.approx(25323.21, abs=0.01)
