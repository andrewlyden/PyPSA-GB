"""Item F24.1: gas carbon on the calorific basis of the efficiencies.

CP30's efficiencies are net-CV (58% CCGT) and its SRMCs (Annex 4, chart CP.25)
use the net-CV factor 0.2016 t/MWh; the historical path keeps the gross-CV 0.183
that pairs with DUKES gross-CV fleet efficiencies.
"""
from pathlib import Path

import pypsa
import pytest
import yaml

from scripts.generators.apply_marginal_costs import (
    CARBON_EMISSION_FACTORS,
    DEFAULT_GAS_EMISSION_FACTOR_BASIS,
    GAS_EMISSION_CARRIERS,
    NET_CV_CARBON_EMISSION_FACTORS,
    apply_marginal_costs_to_network,
    calculate_marginal_cost,
    carbon_emission_factors,
    get_carbon_price,
    get_future_cost_assumptions,
    load_fuel_prices,
)
from scripts.utilities.carrier_definitions import (
    GAS_CO2_T_PER_MWH_TH_GROSS_CV,
    GAS_CO2_T_PER_MWH_TH_NET_CV,
    get_carrier_definitions,
)
from config.config_loader import load_defaults, load_layer

ROOT = Path(__file__).resolve().parents[2]
# Annex 4: 100 p/therm and 147 GBP/t (129 UK ETS + 18 CPS).
GAS_100P_THERM = 34.121425


def _cp30_like(basis='net_cv', heat_credit=False):
    mc = {
        'future_costs': {
            'enabled': True,
            'case': 'cp',
            'price_year': 2024,
            'currency': 'GBP',
            'efficiency_defaults': {'CCGT': 0.52, 'CHP': 0.42, 'CCS Gas': 0.50,
                                    'OCGT': 0.38, 'gas_engine': 0.35},
            'unit_efficiency_overrides': {'ccgt_58': 0.58, 'ccgt_42': 0.42},
            'cases': {'cp': {
                'fuel_prices': {'gas': GAS_100P_THERM},
                'carbon': {'uk_ets': 129.0, 'carbon_price_support': 18.0},
                'variable_om': {'CCGT': 2.0, 'OCGT': 3.0, 'gas_engine': 3.0, 'CCS Gas': 15.0},
            }},
        },
    }
    if basis is not None:
        mc['future_costs']['gas_emission_factor_basis'] = basis
    if heat_credit:
        mc['byproduct_adjustments'] = {'CHP': {
            'mode': 'heat_credit', 'heat_fuel': 'gas', 'heat_to_power': 1.1,
            'boiler_efficiency': 0.85}}
    return {'modelled_year': 2030, 'marginal_costs': mc}


def _apply(scenario, units):
    network = pypsa.Network()
    network.add('Bus', 'GB')
    for name, carrier in units.items():
        network.add('Generator', name, bus='GB', carrier=carrier, p_nom=100.0, efficiency=0.5)
    return apply_marginal_costs_to_network(
        network, load_fuel_prices(scenario), get_carbon_price(scenario), scenario
    ).set_index('generator')


def test_net_cv_reproduces_cp30_cp25_ccgt_ladder():
    result = _apply(_cp30_like(), {'ccgt_58': 'CCGT', 'ccgt_42': 'CCGT'})
    # CP.25: 112 at 58%, 155 at 42%. Fuel / eff + 0.2016 x 147 / eff + VOM 2.
    assert result.at['ccgt_58', 'marginal_cost'] == pytest.approx(112.0, abs=1.0)
    assert result.at['ccgt_42', 'marginal_cost'] == pytest.approx(155.0, abs=1.5)
    assert result.at['ccgt_58', 'emission_factor_kgco2_per_mwh_th'] == pytest.approx(201.6)
    assert result.at['ccgt_58', 'carbon_cost_gbp_per_mwh'] == pytest.approx(0.2016 * 147.0 / 0.58)


def test_gross_cv_misses_the_cp30_ladder_by_the_carbon_gap():
    result = _apply(_cp30_like(basis='gross_cv'), {'ccgt_58': 'CCGT'})
    gap = (0.2016 - 0.183) * 147.0 / 0.58
    assert result.at['ccgt_58', 'marginal_cost'] == pytest.approx(
        112.0 - gap, abs=1.0)
    assert gap > 4.0


def test_net_cv_applies_to_every_gas_carrier_and_ccs_keeps_ten_percent():
    for carrier in GAS_EMISSION_CARRIERS:
        assert NET_CV_CARBON_EMISSION_FACTORS[carrier] == pytest.approx(201.6)
        assert CARBON_EMISSION_FACTORS[carrier] == pytest.approx(183.0)
    assert NET_CV_CARBON_EMISSION_FACTORS['CCS Gas'] == pytest.approx(20.16)
    assert CARBON_EMISSION_FACTORS['CCS Gas'] == pytest.approx(18.3)
    # Non-gas carriers are the same in both tables.
    for carrier in ('coal', 'oil', 'waste', 'biomass', 'nuclear'):
        assert NET_CV_CARBON_EMISSION_FACTORS[carrier] == CARBON_EMISSION_FACTORS[carrier]

    result = _apply(_cp30_like(), {'ocgt': 'OCGT', 'recip': 'gas_engine', 'ccs': 'CCS Gas'})
    assert result.at['ocgt', 'carbon_cost_gbp_per_mwh'] == pytest.approx(0.2016 * 147.0 / 0.38)
    assert result.at['recip', 'carbon_cost_gbp_per_mwh'] == pytest.approx(0.2016 * 147.0 / 0.35)
    assert result.at['ccs', 'carbon_cost_gbp_per_mwh'] == pytest.approx(0.02016 * 147.0 / 0.50)


def test_chp_heat_credit_carbon_is_on_the_unit_basis():
    result = _apply(_cp30_like(heat_credit=True), {'chp': 'CHP'})
    credit = 1.1 * (GAS_100P_THERM + 0.2016 * 147.0) / 0.85
    assert result.at['chp', 'heat_credit_gbp_per_mwh'] == pytest.approx(credit)
    # nationalised_institutions.heat_credit_rates splits the credit with the
    # unit's own carbon per MWh of fuel; the two must agree.
    unit_carbon_per_fuel = result.at['chp', 'carbon_cost_gbp_per_mwh'] * result.at['chp', 'efficiency_used']
    assert unit_carbon_per_fuel == pytest.approx(0.2016 * 147.0)


def test_default_basis_is_gross_and_historical_path_unchanged():
    assumptions = get_future_cost_assumptions(_cp30_like(basis=None))
    assert assumptions['gas_emission_factor_basis'] == DEFAULT_GAS_EMISSION_FACTOR_BASIS == 'gross_cv'
    assert carbon_emission_factors(assumptions) is CARBON_EMISSION_FACTORS
    # Historical scenario (no future-cost case): gross-CV, as calibrated before.
    historical = {'modelled_year': 2020, 'marginal_costs': _cp30_like()['marginal_costs']}
    assert get_future_cost_assumptions(historical) == {}
    assert carbon_emission_factors({}) is CARBON_EMISSION_FACTORS
    _, carbon, _ = calculate_marginal_cost('CCGT', 20.0, 50.0, 0.49)
    assert carbon == pytest.approx(0.183 * 50.0 / 0.49)


def test_unknown_basis_is_refused():
    with pytest.raises(ValueError, match='gas_emission_factor_basis'):
        carbon_emission_factors({'case': 'x', 'gas_emission_factor_basis': 'hhv'})


def test_reported_gas_emissions_use_the_dispatch_constant():
    carriers = get_carrier_definitions()
    for carrier in ('CCGT', 'OCGT', 'CHP', 'micro_CHP', 'gas_engine', 'gas_reciprocating'):
        assert carriers.at[carrier, 'co2_emissions'] == pytest.approx(GAS_CO2_T_PER_MWH_TH_NET_CV)
        if carrier in NET_CV_CARBON_EMISSION_FACTORS:
            assert NET_CV_CARBON_EMISSION_FACTORS[carrier] == pytest.approx(
                carriers.at[carrier, 'co2_emissions'] * 1000.0)
    assert carriers.at['CCS Gas', 'co2_emissions'] * 1000.0 == pytest.approx(
        NET_CV_CARBON_EMISSION_FACTORS['CCS Gas'])
    assert GAS_CO2_T_PER_MWH_TH_GROSS_CV * 1000.0 == pytest.approx(CARBON_EMISSION_FACTORS['CCGT'])


def test_defaults_yaml_matches_code_default_and_cp30_is_net():
    defaults = load_defaults()
    assert (defaults['marginal_costs']['future_costs']['gas_emission_factor_basis']
            == DEFAULT_GAS_EMISSION_FACTOR_BASIS)
    assert (load_layer('source', 'CP30')['marginal_costs']['future_costs']['gas_emission_factor_basis']
            == 'net_cv')
