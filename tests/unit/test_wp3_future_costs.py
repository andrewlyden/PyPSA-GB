import logging
from unittest.mock import mock_open, patch

import pandas as pd
import pypsa
import pytest

from scripts.generators.apply_marginal_costs import (
    CARBON_EMISSION_FACTORS,
    _compute_carrier_factors,
    apply_marginal_costs_to_network,
    calculate_marginal_cost,
    get_carbon_price,
    get_future_cost_assumptions,
    load_fuel_prices,
)
from scripts.generators.thermal_availability import apply_thermal_outage_availability
from scripts.generators.thermal_ramping import apply_thermal_ramp_limits


def _scenario(case='central'):
    return {
        'modelled_year': 2030,
        'marginal_costs': {
            'future_costs': {
                'enabled': True,
                'case': case,
                'price_year': 2024,
                'currency': 'GBP',
                'efficiency_defaults': {
                    'CCGT': 0.52,
                    'CHP': 0.42,
                    'CCS Gas': 0.50,
                },
                'unit_efficiency_overrides': {'unit_override': 0.60},
                'cases': {
                    'low': {
                        'fuel_prices': {'gas': 20.0, 'waste': 10.0},
                        'carbon': {'uk_ets': 35.0, 'carbon_price_support': 18.0},
                        'variable_om': {'CCGT': 1.0},
                    },
                    'central': {
                        'fuel_prices': {'gas': 30.0, 'waste': 15.0},
                        'carbon': {'uk_ets': 55.0, 'carbon_price_support': 18.0},
                        'variable_om': {'CCGT': 2.0, 'CCS Gas': 15.0},
                    },
                    'high': {
                        'fuel_prices': {'gas': 45.0, 'waste': 25.0},
                        'carbon': {'uk_ets': 80.0, 'carbon_price_support': 18.0},
                        'variable_om': {'CCGT': 3.0},
                    },
                },
            },
            'carrier_correction_factors': {
                'enabled': True,
                'allow_future_historical_factors': False,
            },
        },
    }


def test_future_cost_case_selects_prices_and_explicit_carbon_components():
    scenario = _scenario('high')
    assumptions = get_future_cost_assumptions(scenario)

    assert assumptions['case'] == 'high'
    assert assumptions['price_year'] == 2024
    assert load_fuel_prices(scenario)['CCGT'] == pytest.approx(45.0)
    assert load_fuel_prices(scenario)['waste_to_energy'] == pytest.approx(25.0)
    assert get_carbon_price(scenario) == pytest.approx(98.0)


def test_neso_case_reads_generated_annex4_resource():
    payload = {
        "source": {"url": "https://www.neso.energy/document/346806/download"},
        "assumptions": {
            "fuel_prices": {"gas": 34.12},
            "carbon": {"uk_ets": 129.0, "carbon_price_support": 18.0},
        },
    }
    scenario = _scenario()
    future_costs = scenario["marginal_costs"]["future_costs"]
    future_costs["case"] = "neso_cp2030"
    future_costs["cases"]["neso_cp2030"] = {
        "source_file": "generated_neso_annex4_costs.json",
        "fallback_case": "central",
    }

    with patch("scripts.generators.apply_marginal_costs.Path") as path_cls, patch(
        "scripts.generators.apply_marginal_costs.json.load", return_value=payload
    ):
        source_path = path_cls.return_value
        source_path.exists.return_value = True
        source_path.open.return_value = mock_open()()
        assumptions = get_future_cost_assumptions(scenario)

    assert assumptions["case"] == "neso_cp2030"
    assert assumptions["fuel_prices"]["gas"] == pytest.approx(34.12)
    assert assumptions["carbon"] == {
        "uk_ets": 129.0,
        "carbon_price_support": 18.0,
    }
    # Values not supplied by Annex 4 retain the declared central fallback.
    assert assumptions["fuel_prices"]["waste"] == pytest.approx(15.0)

def test_future_scenario_does_not_apply_historical_correction_factors():
    daily = pd.DataFrame(
        {'generator': ['g'], 'carrier': ['CCGT'], 'empirical_mc': [50.0]}
    )
    assert _compute_carrier_factors(daily, {'CCGT': 30.0}, 73.0, _scenario()) == {}


def test_future_cost_application_uses_class_heat_rate_and_variable_om():
    network = pypsa.Network()
    network.add('Bus', 'GB')
    # The network value is a generic placeholder; the explicit future class
    # assumption must take precedence.
    network.add(
        'Generator', 'ccgt', bus='GB', carrier='CCGT', p_nom=100.0,
        efficiency=0.50,
    )
    scenario = _scenario()

    result = apply_marginal_costs_to_network(
        network, load_fuel_prices(scenario), get_carbon_price(scenario), scenario
    ).set_index('generator')

    expected = (30.0 / 0.52) + (183.0 * 73.0 / 1000.0 / 0.52) + 2.0
    assert result.at['ccgt', 'marginal_cost'] == pytest.approx(expected)
    assert result.at['ccgt', 'efficiency_used'] == pytest.approx(0.52)
    assert result.at[
        'ccgt', 'emission_factor_kgco2_per_mwh_th'
    ] == pytest.approx(183.0)
    assert result.at['ccgt', 'carbon_uk_ets_gbp_per_tco2'] == pytest.approx(55.0)
    assert result.at['ccgt', 'carbon_price_support_gbp_per_tco2'] == pytest.approx(18.0)
    assert result.at['ccgt', 'source'] == 'formula+vom'


def test_future_unit_efficiency_override_beats_class_and_network_values():
    network = pypsa.Network()
    network.add('Bus', 'GB')
    network.add(
        'Generator', 'unit_override', bus='GB', carrier='CCGT', p_nom=100.0,
        efficiency=0.45,
    )
    scenario = _scenario()

    result = apply_marginal_costs_to_network(
        network, load_fuel_prices(scenario), get_carbon_price(scenario), scenario
    ).set_index('generator')

    assert result.at['unit_override', 'efficiency_used'] == pytest.approx(0.60)


def test_ccgt_carbon_cost_uses_fuel_input_factor_once():
    fuel, carbon, total = calculate_marginal_cost(
        'CCGT', fuel_price=30.0, carbon_price=73.0, efficiency=0.52
    )

    assert CARBON_EMISSION_FACTORS['CCGT'] == pytest.approx(183.0)
    assert fuel == pytest.approx(30.0 / 0.52)
    assert carbon == pytest.approx(0.183 * 73.0 / 0.52)
    assert total == pytest.approx(fuel + carbon)


def test_thermal_outages_are_reproducible_and_do_not_add_uc_fields():
    snapshots = pd.date_range('2030-01-01', periods=8760, freq='h')
    config = {
        'enabled': True,
        'seed': 7,
        'carriers': {
            'CCGT': {
                'planned_outages_per_year': 1,
                'planned_outage_duration_hours': 24,
                'planned_outage_months': [4],
                'forced_outage_rate': 0.0,
            }
        },
    }
    first = pypsa.Network()
    first.set_snapshots(snapshots)
    first.add('Bus', 'GB')
    first.add('Generator', 'ccgt', bus='GB', carrier='CCGT', p_nom=100.0)
    second = first.copy()

    apply_thermal_outage_availability(first, config, logging.getLogger(__name__))
    apply_thermal_outage_availability(second, config, logging.getLogger(__name__))

    pd.testing.assert_series_equal(
        first.generators_t.p_max_pu['ccgt'], second.generators_t.p_max_pu['ccgt']
    )
    assert first.generators_t.p_max_pu['ccgt'].eq(0.0).sum() == 24
    assert not first.generators.committable.any()


def test_lp_ramp_limits_do_not_enable_unit_commitment():
    network = pypsa.Network()
    snapshots = pd.date_range('2030-01-01', periods=2, freq='h')
    network.set_snapshots(snapshots)
    network.add('Bus', 'GB')
    network.add('Load', 'demand', bus='GB', p_set=[0.0, 100.0])
    network.add('Generator', 'ccgt', bus='GB', carrier='CCGT', p_nom=100.0, marginal_cost=1.0)
    network.add('Generator', 'backup', bus='GB', carrier='backup', p_nom=100.0, marginal_cost=100.0)

    applied = apply_thermal_ramp_limits(
        network,
        {'enabled': True, 'carriers': {'CCGT': {'ramp_limit_up': 0.5, 'ramp_limit_down': 0.4}}},
        logging.getLogger(__name__),
    )

    assert applied == 1
    assert network.generators.at['ccgt', 'ramp_limit_up'] == pytest.approx(0.5)
    assert network.generators.at['ccgt', 'ramp_limit_down'] == pytest.approx(0.4)
    assert not network.generators.at['ccgt', 'committable']

    # Tighten the rate to 40 MW/h and verify the LP uses expensive backup.
    network.generators.at['ccgt', 'ramp_limit_up'] = 0.4
    status, condition = network.optimize(solver_name='highs')
    assert status == 'ok', condition
    assert network.generators_t.p.at[snapshots[1], 'ccgt'] == pytest.approx(40.0)
    assert network.generators_t.p.at[snapshots[1], 'backup'] == pytest.approx(60.0)
