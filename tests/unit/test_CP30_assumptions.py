from pathlib import Path
import logging

import pypsa
import pytest

from scripts.CP30.assumptions import (
    get_cp30_marginal_cost_overrides,
    load_cp30_grid_connected_electrolysis_capacity_mw,
    load_or_derive_cp30_electrolysis_capacity_mw,
)
from scripts.generators.apply_marginal_costs import apply_marginal_costs_to_network
from scripts.hydrogen.add_hydrogen_system import add_hydrogen_system


CP30_ED1 = Path("resources/CP30/CP30_ED1.csv")
CP30_ES1 = Path("resources/CP30/CP30_ES1.csv")


def test_cp30_marginal_cost_overrides_are_explicit_and_cp30_only():
    assert get_cp30_marginal_cost_overrides({}) == {}

    overrides = get_cp30_marginal_cost_overrides({"scenario_source": "CP30"})
    assert overrides["CCS Gas"] == pytest.approx(100.0)
    assert overrides["CCS Biomass"] == pytest.approx(120.0)
    assert overrides["Hydrogen"] == pytest.approx(80.0)


@pytest.mark.skipif(not CP30_ED1.exists(), reason="CP30 ED1 resource not available")
def test_cp30_electrolysis_capacity_prefers_ed1_installed_capacity():
    capacity = load_cp30_grid_connected_electrolysis_capacity_mw(
        CP30_ED1,
        "Further Flex and Renewables",
        2030,
    )

    assert capacity == pytest.approx(3653.121)


@pytest.mark.skipif(not CP30_ED1.exists() or not CP30_ES1.exists(), reason="CP30 resources not available")
def test_cp30_electrolysis_capacity_falls_back_to_cp30_es1_when_ed1_missing():
    capacity, source = load_or_derive_cp30_electrolysis_capacity_mw(
        CP30_ED1,
        CP30_ES1,
        "New Dispatch",
        2030,
        full_load_hours=4000.0,
    )

    assert source == "CP30_ES1_electrolysis_demand_derived"
    assert capacity == pytest.approx(3347.5225)


def test_apply_marginal_costs_uses_cp30_clean_dispatchable_assumptions():
    network = pypsa.Network()
    network.add("Bus", "bus")
    network.add("Generator", "ccs_gas", bus="bus", carrier="CCS Gas", p_nom=100.0)
    network.add("Generator", "ccs_bio", bus="bus", carrier="CCS Biomass", p_nom=100.0)
    network.add("Generator", "h2", bus="bus", carrier="Hydrogen", p_nom=100.0)

    mc = apply_marginal_costs_to_network(
        network,
        fuel_prices={},
        carbon_price=0.0,
        scenario_config={"scenario_source": "CP30", "modelled_year": 2030},
    )

    by_generator = mc.set_index("generator")
    assert by_generator.loc["ccs_gas", "marginal_cost"] == pytest.approx(100.0)
    assert by_generator.loc["ccs_bio", "marginal_cost"] == pytest.approx(120.0)
    assert by_generator.loc["h2", "marginal_cost"] == pytest.approx(80.0)
    assert set(by_generator["source"]) == {"cp30_assumption"}


def test_hydrogen_system_accepts_cp30_hydrogen_carrier():
    network = pypsa.Network()
    network.add("Bus", "bus", x=0.0, y=0.0)
    network.add("Generator", "cp30_h2", bus="bus", carrier="Hydrogen", p_nom=100.0)

    result = add_hydrogen_system(
        network,
        electrolysis_capacity_mw=200.0,
        h2_generation_capacity_mw=100.0,
        logger=logging.getLogger("test_cp30_hydrogen"),
    )

    assert "cp30_h2" not in result.generators.index
    assert result.links.loc["H2_turbine_cp30_h2", "carrier"] == "H2_turbine"
    assert result.links[result.links.carrier == "electrolysis"].p_nom.sum() == pytest.approx(200.0)
    # The generator's 100 MW is electrical: the link takes 200 MW of hydrogen at 50%.
    turbine = result.links.loc["H2_turbine_cp30_h2"]
    assert turbine.p_nom * turbine.efficiency == pytest.approx(100.0)
    # A week of full output needs 100 / 0.5 x 168 MWh of hydrogen.
    assert result.stores.loc["GB_H2_storage", "e_nom"] == pytest.approx(100.0 / 0.5 * 168)


def test_cp30_validation_checks_hydrogen_on_electrical_output():
    import pandas as pd

    from scripts.CP30.validation import _actual_link_output_capacity_by_carrier

    links = pd.DataFrame({"carrier": ["H2_turbine", "electrolysis"], "p_nom": [552.0, 1000.0],
                          "efficiency": [0.5, 0.7]})
    assert _actual_link_output_capacity_by_carrier(links, "H2_turbine") == pytest.approx(276.0)