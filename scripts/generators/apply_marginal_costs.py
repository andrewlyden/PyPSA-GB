"""
Apply marginal costs to generators in PyPSA network.

Simplified two-tier approach:
  1. Empirical MCs (from ELEXON calibration) when available
  2. Formula/subsidy fallback otherwise

Flat if/elif logic — no layers, no static prerun files.

Fixes (November 2026), each marked "FIX-n" in the code
------------------------------------------------------------------
FIX-1         ``calculate_marginal_cost`` silently replaced an efficiency
              outside (0, 1] (or NaN) with 1.0, under-stating MC. It now raises
              ``ValueError``, like ``cost_utils.marginal_cost``.
FIX-2         The carrier-factor efficiencies are a module constant
              ``FORMULA_CARRIER_EFF`` (was a dict inside
              ``_compute_carrier_factors``) so the thermal calibration and the
              correction-factor exporter import them instead of copying them.
              ``biomass`` (lower case, the network carrier) was missing, so
              biomass never got a factor; it is added with the Biomass value.
"""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
from scripts.utilities.logging_config import setup_logging
from scripts.utilities.network_io import load_network, save_network
from scripts.CP30.assumptions import get_cp30_marginal_cost_overrides
from scripts.utilities.carrier_definitions import (
    CCS_GAS_CAPTURE_RATE,
    GAS_CO2_T_PER_MWH_TH_GROSS_CV,
    GAS_CO2_T_PER_MWH_TH_NET_CV,
)

logger = setup_logging("apply_marginal_costs")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

PROTECTED_CARRIERS = {
    'load_shedding', 'load shedding', 'voll', 'VOLL',
    'demand response', 'demand_response',
}

THERMAL_CARRIERS = {
    'CCGT', 'OCGT', 'Coal', 'coal', 'Oil', 'oil',
    'Biomass', 'biomass', 'Bioenergy', 'CHP', 'micro_CHP',
    'gas_engine', 'waste', 'Waste', 'waste_to_energy',
    'landfill_gas', 'sewage_gas', 'biogas', 'advanced_biofuel',
    'Conventional Steam', 'Conventional steam',
    'Biomass (dedicated)', 'Biomass (co-firing)',
    'CCS Gas', 'CCS Biomass', 'Hydrogen', 'H2',
}

RENEWABLE_CARRIERS = {
    'wind_onshore', 'wind_offshore', 'solar_pv',
    'large_hydro', 'small_hydro', 'tidal_stream',
    'shoreline_wave', 'tidal_lagoon', 'marine',
}

# Biofuel carriers eligible for ROC subsidy deduction from their thermal MC.
# Dedicated biomass receives 1.5 ROC/MWh under Ofgem banding; biogas/landfill/
# sewage gas and advanced biofuels may also hold ROC accreditation.
# Waste-to-energy is excluded (gate-fee economics differ from ROC subsidy).
BIOFUEL_CARRIERS = {
    'biomass', 'Biomass', 'Biomass (dedicated)', 'Biomass (co-firing)',
    'Bioenergy', 'advanced_biofuel', 'biogas',
    'landfill_gas', 'sewage_gas',
}



# Direct combustion emission factors (kg CO2/MWh_thermal fuel input).
#
# These are fuel-input factors because the marginal-cost calculation divides
# fuel and carbon inputs by electrical efficiency. The previous 488 kgCO2/MWh
# gas value was an electricity-output/lifecycle intensity and was therefore
# divided by efficiency a second time. Natural gas is the UK Government
# gross-CV factor of about 0.183 kgCO2/kWh_th, which pairs with the gross-CV
# fleet efficiencies (DUKES) of the historical path. Coal and oil use
# corresponding direct-combustion factors; biogenic CO2 is zero for dispatch
# carbon pricing. CCS Gas assumes 90% capture of the unabated combustion factor.
# A future-cost case on net-CV efficiencies prices gas on the net-CV factor
# instead (``carbon_emission_factors``; item F24.1).
# Efficiencies used to compute carrier correction factors (empirical / formula).
# FIX-2: a module constant so other scripts use the same values.
FORMULA_CARRIER_EFF = {
    'CCGT': 0.49, 'OCGT': 0.35, 'Coal': 0.36, 'coal': 0.36,
    'Oil': 0.30, 'oil': 0.30, 'Biomass': 0.35, 'biomass': 0.35, 'nuclear': 0.33,
}

GAS_EMISSION_CARRIERS = ('gas', 'Gas', 'CCGT', 'OCGT', 'CHP', 'micro_CHP', 'gas_engine')
_GAS_GROSS_KG = round(GAS_CO2_T_PER_MWH_TH_GROSS_CV * 1000.0, 6)
CARBON_EMISSION_FACTORS = {
    'coal': 341, 'Coal': 341, 'Conventional Steam': 341, 'Conventional steam': 341,
    **{carrier: _GAS_GROSS_KG for carrier in GAS_EMISSION_CARRIERS},
    'oil': 267, 'Oil': 267,
    'biomass': 0, 'Biomass': 0, 'Biomass (dedicated)': 0, 'Biomass (co-firing)': 0,
    'Bioenergy': 0, 'advanced_biofuel': 0,
    'waste': 180, 'Waste': 180, 'waste_to_energy': 180,
    'landfill_gas': 0, 'sewage_gas': 0, 'biogas': 0,
    'CCS Gas': round(_GAS_GROSS_KG * (1.0 - CCS_GAS_CAPTURE_RATE), 6), 'CCS Biomass': 0,
    'nuclear': 12, 'Nuclear': 12, 'PWR': 12, 'AGR': 12,
    'hydro': 24, 'Hydro': 24, 'Large Hydro': 24, 'large_hydro': 24,
    'Small Hydro': 24, 'small_hydro': 24, 'Hydro / pumped storage': 24,
    'Pumped Storage': 0, 'Battery': 0, 'battery': 0,
    'wind': 11, 'Wind': 11, 'wind_onshore': 11, 'wind_offshore': 11,
    'Wind (Onshore)': 11, 'Wind (Offshore)': 11,
    'solar': 48, 'Solar': 48, 'solar_pv': 48,
    'geothermal': 38, 'tidal_stream': 15, 'shoreline_wave': 15,
    'H2': 0, 'Hydrogen': 0, 'fuel_cell': 0,
}

# Gas on CP30's net-CV basis (0.2016 t/MWh, NESO Clean Power 2030 Annex 4 chart
# CP.25), for a future-cost case whose efficiencies are net-CV. The same constant
# sets the gas carriers' reported co2_emissions (carrier_definitions.py).
_GAS_NET_KG = round(GAS_CO2_T_PER_MWH_TH_NET_CV * 1000.0, 6)
NET_CV_CARBON_EMISSION_FACTORS = {
    **CARBON_EMISSION_FACTORS,
    **{carrier: _GAS_NET_KG for carrier in GAS_EMISSION_CARRIERS},
    'CCS Gas': round(_GAS_NET_KG * (1.0 - CCS_GAS_CAPTURE_RATE), 6),
}

# Calorific basis of the gas emission factor in a future-cost case
# (future_costs.gas_emission_factor_basis). Must match config/defaults/base/marginal_costs.yaml.
DEFAULT_GAS_EMISSION_FACTOR_BASIS = 'gross_cv'
#: MWh (gross calorific value) per therm: the conversion NESO's Annex 4 extract uses
#: (resources/market/neso_clean_power_2030_annex4_costs.json, therm_to_mwh).
THERM_TO_MWH = 0.0293071
#: Code default of marginal_costs.future_costs.gas_price_override_pence_per_therm
#: (null: the case's own gas price). Must match config/defaults/base/marginal_costs.yaml.
DEFAULT_GAS_PRICE_OVERRIDE_PENCE_PER_THERM = None
_GAS_OVERRIDE_LOGGED: set = set()
_EMISSION_FACTORS_BY_BASIS = {
    'gross_cv': CARBON_EMISSION_FACTORS,
    'net_cv': NET_CV_CARBON_EMISSION_FACTORS,
}


def carbon_emission_factors(future_assumptions: dict | None) -> dict:
    """Emission-factor table (kg CO2/MWh_th) for the scenario's cost path.

    The historical path, and a future-cost case on the default basis, keep the
    gross-CV gas factor; a case declaring ``gas_emission_factor_basis: net_cv``
    (CP30, whose efficiencies are net-CV: 58% CCGT) prices gas at 201.6 kg and
    CCS Gas at 10% of that.
    """
    if not future_assumptions:
        return CARBON_EMISSION_FACTORS
    basis = str(future_assumptions.get('gas_emission_factor_basis',
                                       DEFAULT_GAS_EMISSION_FACTOR_BASIS)).lower()
    if basis not in _EMISSION_FACTORS_BY_BASIS:
        raise ValueError(
            f"Unknown gas_emission_factor_basis {basis!r}; expected one of "
            f"{sorted(_EMISSION_FACTORS_BY_BASIS)}"
        )
    return _EMISSION_FACTORS_BY_BASIS[basis]


# ══════════════════════════════════════════════════════════════════════════════
# HISTORICAL FUEL AND CARBON PRICES (2010-2024)
# ══════════════════════════════════════════════════════════════════════════════

HISTORICAL_FUEL_PRICES = {
    2010: {'gas': 20.0, 'coal': 10.0, 'oil': 45.0, 'biomass': 35.0},
    2011: {'gas': 22.0, 'coal': 12.0, 'oil': 50.0, 'biomass': 35.0},
    2012: {'gas': 25.0, 'coal': 12.0, 'oil': 55.0, 'biomass': 35.0},
    2013: {'gas': 27.0, 'coal': 10.0, 'oil': 55.0, 'biomass': 35.0},
    2014: {'gas': 22.0, 'coal': 8.0, 'oil': 50.0, 'biomass': 35.0},
    2015: {'gas': 17.0, 'coal': 7.0, 'oil': 35.0, 'biomass': 35.0},
    2016: {'gas': 15.0, 'coal': 8.0, 'oil': 30.0, 'biomass': 35.0},
    2017: {'gas': 17.0, 'coal': 10.0, 'oil': 35.0, 'biomass': 35.0},
    2018: {'gas': 25.0, 'coal': 12.0, 'oil': 45.0, 'biomass': 35.0},
    2019: {'gas': 20.0, 'coal': 8.0, 'oil': 40.0, 'biomass': 35.0},
    2020: {'gas': 12.0, 'coal': 6.0, 'oil': 25.0, 'biomass': 35.0},
    2021: {'gas': 45.0, 'coal': 15.0, 'oil': 50.0, 'biomass': 35.0},
    2022: {'gas': 80.0, 'coal': 25.0, 'oil': 70.0, 'biomass': 40.0},
    2023: {'gas': 40.0, 'coal': 15.0, 'oil': 55.0, 'biomass': 40.0},
    2024: {'gas': 35.0, 'coal': 12.0, 'oil': 50.0, 'biomass': 40.0},
}

HISTORICAL_CARBON_PRICES = {
    2010: 12.0, 2011: 10.0, 2012: 7.0, 2013: 21.0, 2014: 21.0,
    2015: 25.0, 2016: 23.0, 2017: 23.0, 2018: 33.0, 2019: 43.0,
    2020: 43.0, 2021: 63.0, 2022: 88.0, 2023: 68.0, 2024: 85.0,
}


def get_future_cost_assumptions(scenario_config: dict | None) -> dict:
    """Return the selected future commodity-cost case, if one is enabled.

    Future cases are deliberately scenario-configured rather than inferred from
    historical calibration. All monetary values must be expressed in the
    declared real-price year.
    """
    scenario_config = scenario_config or {}
    modelled_year = scenario_config.get('modelled_year')
    if not modelled_year or modelled_year <= 2024:
        return {}

    mc_config = scenario_config.get('marginal_costs', {})
    future_cfg = mc_config.get('future_costs', {})
    if not future_cfg.get('enabled', False):
        return {}

    cases = future_cfg.get('cases', {})
    case_name = str(future_cfg.get('case', 'central')).lower()
    if case_name not in cases:
        raise ValueError(
            f"Unknown future marginal-cost case '{case_name}'. "
            f"Available cases: {sorted(cases)}"
        )

    configured_case = deepcopy(cases[case_name])
    source_file = configured_case.pop('source_file', None)
    fallback_case = configured_case.pop('fallback_case', None)

    if source_file:
        if not fallback_case or fallback_case not in cases:
            raise ValueError(
                f"Future marginal-cost case '{case_name}' uses source_file but "
                "does not name a valid fallback_case."
            )
        selected = deepcopy(cases[fallback_case])
        source_path = Path(source_file)
        if not source_path.exists():
            raise FileNotFoundError(
                f"NESO future-cost source file is missing: {source_path}. "
                "Generate it with scripts/market/fetch_neso_cp2030_annex4_costs.py."
            )
        with source_path.open(encoding='utf-8') as handle:
            payload = json.load(handle)

        sourced = payload.get('assumptions', {})
        fuel_prices = sourced.get('fuel_prices', {})
        carbon = sourced.get('carbon', {})
        required_carbon = {'uk_ets', 'carbon_price_support'}
        if 'gas' not in fuel_prices or not required_carbon.issubset(carbon):
            raise ValueError(
                f"Invalid sourced future-cost file {source_path}: expected gas "
                "fuel price plus UK ETS and Carbon Price Support values."
            )
        selected.setdefault('fuel_prices', {}).update(fuel_prices)
        selected.setdefault('carbon', {}).update(carbon)
        selected['source_file'] = str(source_path)
        selected['source_provenance'] = payload.get('source', {})
    else:
        selected = configured_case

    selected['case'] = case_name
    selected['price_year'] = int(future_cfg.get('price_year', 2024))
    selected['currency'] = str(future_cfg.get('currency', 'GBP'))
    selected['efficiency_defaults'] = future_cfg.get('efficiency_defaults', {})
    selected['unit_efficiency_overrides'] = future_cfg.get(
        'unit_efficiency_overrides', {}
    )
    selected['gas_emission_factor_basis'] = str(future_cfg.get(
        'gas_emission_factor_basis', DEFAULT_GAS_EMISSION_FACTOR_BASIS
    ))
    _apply_carbon_price_support_end(selected, future_cfg, scenario_config)
    _apply_gas_price_override(selected, future_cfg)
    return selected


def _apply_gas_price_override(selected: dict, future_cfg: dict) -> None:
    """Replace the case's gas price by ``future_costs.gas_price_override_pence_per_therm`` (null: keep it).

    For CP30 (16 Oct 2026, NESO review S1-04): Annex 4's text gives "around 100p/therm",
    which the neso_cp2030 extract carries (34.12 GBP/MWh), while CP30's assumptions
    (CP.22, Clean pathways) use 100.68 p/therm, 34.354 GBP/MWh, the price CP.25's SRMCs,
    the hydrogen price and the foreign-price rebase already use. Recorded on the case.
    """
    value = future_cfg.get('gas_price_override_pence_per_therm', DEFAULT_GAS_PRICE_OVERRIDE_PENCE_PER_THERM)
    if value is None:
        return
    pence = float(value)
    if not 0.0 < pence < 1000.0:
        raise ValueError(f"future_costs.gas_price_override_pence_per_therm must be a positive p/therm, not {value}")
    fuel = selected.setdefault('fuel_prices', {})
    selected['gas_price_before_override_gbp_per_mwh'] = fuel.get('gas')
    fuel['gas'] = pence / 100.0 / THERM_TO_MWH
    selected['gas_price_override_pence_per_therm'] = pence
    if pence in _GAS_OVERRIDE_LOGGED:   # resolved once per generator; log it once
        return
    _GAS_OVERRIDE_LOGGED.add(pence)
    logger.info("Gas price override: %.2f p/therm = %.3f GBP/MWh thermal (case %s gave %s)", pence, fuel['gas'],
                selected.get('case'), selected['gas_price_before_override_gbp_per_mwh'])


def _modelled_period_start(scenario_config: dict) -> pd.Timestamp:
    """First instant the scenario models: the solve period start, else 1 January of the modelled year."""
    period = scenario_config.get('solve_period') or {}
    if period.get('enabled') and period.get('start'):
        return pd.Timestamp(period['start'])
    return pd.Timestamp(year=int(scenario_config['modelled_year']), month=1, day=1)


def _apply_carbon_price_support_end(selected: dict, future_cfg: dict, scenario_config: dict) -> None:
    """Drop Carbon Price Support from a case whose modelled period starts on or after its end date.

    ``future_costs.carbon_price_support_end`` (null = CPS continues). HM Treasury
    (written statement HLWS1528, 16 April 2026) discontinues CPS for power
    generation from April 2028, leaving the UK ETS as the only carbon price. A
    sourced case such as neso_cp2030 carries CPS inside its total; the UK ETS part
    is kept as the case gives it. Recorded on the case so the marginal-cost
    breakdown (and the Exchequer ledger that reads it) shows CPS at zero.
    """
    end = future_cfg.get('carbon_price_support_end')
    if not end:
        return
    carbon = selected.setdefault('carbon', {})
    if _modelled_period_start(scenario_config) >= pd.Timestamp(end):
        removed = float(carbon.get('carbon_price_support', 0.0))
        carbon['carbon_price_support'] = 0.0
        selected['carbon_price_support_ended'] = str(end)
        logger.info(
            "Carbon Price Support ends %s: removed %.2f GBP/tCO2 from the %s case; "
            "UK ETS %.2f GBP/tCO2 kept", end, removed, selected.get('case'),
            float(carbon.get('uk_ets', 0.0)),
        )


def _future_carbon_components(assumptions: dict) -> tuple[float, float]:
    """Return UK ETS and Carbon Price Support assumptions in GBP/tCO2."""
    carbon = assumptions.get('carbon', {})
    return (
        float(carbon.get('uk_ets', 0.0)),
        float(carbon.get('carbon_price_support', 0.0)),
    )


def _lookup_carrier_value(values: dict, carrier: str, default: float = 0.0) -> float:
    """Look up a carrier setting while tolerating case-only differences."""
    if carrier in values:
        return float(values[carrier])
    carrier_lower = str(carrier).lower()
    for key, value in values.items():
        if str(key).lower() == carrier_lower:
            return float(value)
    return float(default)


def resolve_thermal_efficiency(gen_name: str, generator: pd.Series,
                               scenario_config: dict | None) -> float:
    """Choose unit data first, then configured class heat-rate assumptions.

    Order: an exact ``unit_efficiency_overrides`` entry, then the generator's
    own ``unit_efficiency`` (set from an evidenced station schedule, e.g. the
    CP30 gas schedule), then the class value, then the network placeholder.
    """
    assumptions = get_future_cost_assumptions(scenario_config)
    if not assumptions:
        value = generator.get('efficiency', 1.0)
        return float(value) if pd.notna(value) else 1.0

    unit_overrides = assumptions.get('unit_efficiency_overrides', {})
    if gen_name in unit_overrides:
        value = float(unit_overrides[gen_name])
        if 0.0 < value <= 1.0:
            return value
        raise ValueError(f"Invalid unit efficiency for {gen_name}: {value}")

    unit_value = pd.to_numeric(generator.get('unit_efficiency', np.nan), errors='coerce')
    if pd.notna(unit_value):
        if 0.0 < float(unit_value) <= 1.0:
            return float(unit_value)
        raise ValueError(f"Invalid unit_efficiency for {gen_name}: {unit_value}")

    carrier = generator.get('carrier', '')
    class_efficiency = _lookup_carrier_value(
        assumptions.get('efficiency_defaults', {}), carrier, default=np.nan
    )
    if pd.notna(class_efficiency):
        if not 0.0 < class_efficiency <= 1.0:
            raise ValueError(
                f'Invalid class efficiency for {carrier}: {class_efficiency}'
            )
        return class_efficiency

    value = generator.get('efficiency', np.nan)
    if pd.notna(value) and 0.0 < float(value) <= 1.0:
        return float(value)
    return 1.0


# ══════════════════════════════════════════════════════════════════════════════
# PRICE LOADING
# ══════════════════════════════════════════════════════════════════════════════

def get_historical_fuel_prices(year: int) -> dict:
    """Get historical fuel prices, clamped to available range."""
    if year in HISTORICAL_FUEL_PRICES:
        return HISTORICAL_FUEL_PRICES[year].copy()
    elif year < min(HISTORICAL_FUEL_PRICES):
        return HISTORICAL_FUEL_PRICES[min(HISTORICAL_FUEL_PRICES)].copy()
    else:
        return HISTORICAL_FUEL_PRICES[max(HISTORICAL_FUEL_PRICES)].copy()


def get_historical_carbon_price(year: int) -> float:
    """Get historical carbon price, clamped to available range."""
    if year in HISTORICAL_CARBON_PRICES:
        return HISTORICAL_CARBON_PRICES[year]
    elif year < min(HISTORICAL_CARBON_PRICES):
        return HISTORICAL_CARBON_PRICES[min(HISTORICAL_CARBON_PRICES)]
    else:
        return HISTORICAL_CARBON_PRICES[max(HISTORICAL_CARBON_PRICES)]


def _expand_fuel_prices(base_prices: dict) -> dict:
    """Expand base fuel prices {gas, coal, oil, biomass} to all carrier variants."""
    gas = base_prices.get('gas', 35.0)
    coal = base_prices.get('coal', 30.0)
    oil = base_prices.get('oil', 75.0)
    bio = base_prices.get('biomass', 45.0)
    waste = base_prices.get('waste', bio * 0.5)
    hydrogen = base_prices.get('hydrogen', base_prices.get('H2', 50.0))
    return {
        'gas': gas, 'Gas': gas, 'CCGT': gas, 'OCGT': gas,
        'CHP': gas, 'micro_CHP': gas, 'gas_engine': gas, 'CCS Gas': gas,
        'coal': coal, 'Coal': coal,
        'Conventional Steam': coal, 'Conventional steam': coal,
        'oil': oil, 'Oil': oil,
        'biomass': bio, 'Biomass': bio,
        'Biomass (dedicated)': bio, 'Biomass (co-firing)': bio,
        'Bioenergy': bio, 'biogas': bio * 0.9,
        'landfill_gas': bio * 0.6, 'sewage_gas': bio * 0.7,
        'advanced_biofuel': bio * 1.1, 'CCS Biomass': bio,
        'waste': waste, 'Waste': waste, 'waste_to_energy': waste,
        'nuclear': 8.0, 'Nuclear': 8.0, 'PWR': 8.0, 'AGR': 8.0,
        'hydro': 5.0, 'Hydro': 5.0, 'Large Hydro': 5.0, 'large_hydro': 5.0,
        'Small Hydro': 5.0, 'small_hydro': 5.0, 'Hydro / pumped storage': 5.0,
        'Pumped Storage': 0.1, 'Battery': 0.1, 'battery': 0.1,
        'wind': 0.5, 'Wind': 0.5, 'wind_onshore': 0.5, 'wind_offshore': 0.5,
        'Wind (Onshore)': 0.5, 'Wind (Offshore)': 0.5,
        'solar': 0.5, 'Solar': 0.5, 'solar_pv': 0.5,
        'geothermal': 3.0, 'tidal_stream': 1.0, 'shoreline_wave': 1.0,
        'H2': hydrogen, 'Hydrogen': hydrogen, 'fuel_cell': hydrogen,
    }


def _get_cost_fes_year(scenario_config: dict):
    """Return the FES year used only for cost assumptions."""
    return (
        scenario_config.get('FES_year')
        or scenario_config.get('fleet', {}).get('spatial_proxy', {}).get('FES_year')
    )

def load_fuel_prices(scenario_config: dict) -> dict:
    """Load fuel prices: historical table > scenario override > FES > defaults."""
    mc_config = scenario_config.get('marginal_costs', {})
    modelled_year = scenario_config.get('modelled_year')

    # Historical scenarios: always use lookup table
    if modelled_year and modelled_year <= 2024:
        prices = get_historical_fuel_prices(modelled_year)
        logger.info(f"Using HISTORICAL fuel prices for {modelled_year}: "
                    f"gas={prices['gas']}, coal={prices['coal']}")
        return _expand_fuel_prices(prices)

    future_assumptions = get_future_cost_assumptions(scenario_config)
    if future_assumptions:
        prices = future_assumptions.get('fuel_prices', {})
        logger.info(
            "Using future %s cost case in %s real %s: gas=%.2f GBP/MWh_th",
            future_assumptions['case'],
            future_assumptions['currency'],
            future_assumptions['price_year'],
            float(prices.get('gas', 0.0)),
        )
        return _expand_fuel_prices(prices)

    # Scenario explicit override
    if 'fuel_prices' in mc_config:
        logger.info("Using scenario-specific fuel prices")
        return _expand_fuel_prices(mc_config['fuel_prices'])

    # FES dynamic prices for future scenarios
    if modelled_year and modelled_year > 2024 and mc_config.get('use_fes_prices', True):
        fes_year = _get_cost_fes_year(scenario_config)
        fuel_file = scenario_config.get('_fuel_price_file')
        if fes_year:
            fes_prices = _load_fes_fuel_prices(fes_year, modelled_year, fuel_file)
            if fes_prices:
                logger.info(f"Using FES fuel prices for {modelled_year} (FES {fes_year})")
                return _expand_fuel_prices(fes_prices)

    # Config defaults
    default_fuels = mc_config.get('fuel_prices', {'gas': 35, 'coal': 30, 'oil': 75, 'biomass': 45})
    logger.info("Using configured DEFAULT fuel prices")
    return _expand_fuel_prices(default_fuels)


def get_carbon_price(scenario_config: dict) -> float:
    """Get carbon price: historical table > scenario override > FES > defaults."""
    mc_config = scenario_config.get('marginal_costs', {})
    modelled_year = scenario_config.get('modelled_year')

    if modelled_year and modelled_year <= 2024:
        price = get_historical_carbon_price(modelled_year)
        logger.info(f"Using HISTORICAL carbon price for {modelled_year}: {price:.2f}/tCO2")
        return price

    future_assumptions = get_future_cost_assumptions(scenario_config)
    if future_assumptions:
        uk_ets, cps = _future_carbon_components(future_assumptions)
        total = uk_ets + cps
        logger.info(
            "Using future %s carbon case: UK ETS=%.2f + CPS=%.2f = %.2f GBP/tCO2 "
            "(%s real %s)",
            future_assumptions['case'], uk_ets, cps, total,
            future_assumptions['currency'], future_assumptions['price_year'],
        )
        return total

    if 'carbon_price' in mc_config:
        return mc_config['carbon_price']

    if modelled_year and modelled_year > 2024 and mc_config.get('use_fes_prices', True):
        fes_year = _get_cost_fes_year(scenario_config)
        carbon_file = scenario_config.get('_carbon_price_file')
        if fes_year:
            fes_carbon = _load_fes_carbon_price(fes_year, modelled_year, carbon_file)
            if fes_carbon is not None:
                return fes_carbon

    return mc_config.get('carbon_price', 85.0)


def _load_fes_fuel_prices(fes_year, modelled_year, fuel_price_file=None):
    """Load FES fuel prices CSV, interpolating for modelled_year."""
    if fuel_price_file is None:
        fuel_price_file = f"resources/marginal_costs/fuel_prices_{fes_year}.csv"
    path = Path(fuel_price_file)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if not {'year', 'fuel', 'price_gbp_per_mwh_thermal'}.issubset(df.columns):
            return None
        prices = {}
        for fuel in df['fuel'].unique():
            fdf = df[df['fuel'] == fuel].sort_values('year')
            exact = fdf[fdf['year'] == modelled_year]
            if len(exact) > 0:
                prices[fuel] = float(exact.iloc[0]['price_gbp_per_mwh_thermal'])
            else:
                prices[fuel] = float(np.interp(modelled_year, fdf['year'], fdf['price_gbp_per_mwh_thermal']))
        logger.info(f"Loaded FES fuel prices for {modelled_year} from FES {fes_year}")
        return prices
    except Exception as e:
        logger.warning(f"Failed to load FES fuel prices: {e}")
        return None


def _load_fes_carbon_price(fes_year, modelled_year, carbon_price_file=None):
    """Load FES carbon price CSV, interpolating for modelled_year."""
    if carbon_price_file is None:
        carbon_price_file = f"resources/marginal_costs/carbon_prices_{fes_year}.csv"
    path = Path(carbon_price_file)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if not {'year', 'carbon_price_gbp_per_tco2'}.issubset(df.columns):
            return None
        df = df.sort_values('year')
        exact = df[df['year'] == modelled_year]
        if len(exact) > 0:
            return float(exact.iloc[0]['carbon_price_gbp_per_tco2'])
        return float(np.interp(modelled_year, df['year'], df['carbon_price_gbp_per_tco2']))
    except Exception as e:
        logger.warning(f"Failed to load FES carbon price: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# FORMULA MC CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

def calculate_marginal_cost(carrier: str, fuel_price: float,
                            carbon_price: float, efficiency: float = 1.0,
                            emission_factors: dict | None = None) -> tuple:
    """Calculate electrical-output fuel and carbon costs in GBP/MWh_e.

    Fuel price is GBP/MWh_th and the emission factor is kgCO2/MWh_th, so both
    inputs are divided by electrical efficiency exactly once. ``emission_factors``
    defaults to the gross-CV table (``carbon_emission_factors`` picks the
    scenario's).

    Raises
    ------
    ValueError
        If efficiency is NaN or outside (0, 1] (FIX-1).
    """
    # FIX-1: fail loudly instead of silently using 1.0
    if not (efficiency == efficiency) or efficiency <= 0 or efficiency > 1.0:
        raise ValueError(
            f"FAILED: calculate_marginal_cost({carrier!r}) - efficiency {efficiency!r} "
            "is not in (0, 1]"
        )
    fuel_cost = fuel_price / efficiency
    if emission_factors is None:
        emission_factors = CARBON_EMISSION_FACTORS
    emission_factor = emission_factors.get(carrier, 0)
    carbon_cost = (emission_factor * carbon_price) / 1000 / efficiency
    return fuel_cost, carbon_cost, fuel_cost + carbon_cost


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_thermal_empirical_mc(filepath):
    """Load daily thermal empirical MCs. Returns DataFrame or None."""
    if not filepath or not Path(filepath).exists():
        return None
    try:
        df = pd.read_csv(filepath)
        if 'generator' not in df.columns or 'empirical_mc' not in df.columns:
            return None
        logger.info(f"Loaded thermal empirical MCs: {df['generator'].nunique()} generators, "
                    f"{df['date'].nunique() if 'date' in df.columns else '?'} days")
        return df
    except Exception as e:
        logger.warning(f"Failed to load thermal empirical MCs: {e}")
        return None


def _load_renewable_empirical_mc(filepath):
    """Load renewable empirical MCs. Returns {generator: mc} dict."""
    if not filepath or not Path(filepath).exists():
        return {}
    try:
        df = pd.read_csv(filepath)
        if 'generator' not in df.columns or 'empirical_mc' not in df.columns:
            return {}
        mcs = dict(zip(df['generator'], df['empirical_mc']))
        logger.info(f"Loaded renewable empirical MCs: {len(mcs)} generators")
        return mcs
    except Exception as e:
        logger.warning(f"Failed to load renewable empirical MCs: {e}")
        return {}


def _get_roc_buyout_price(buyout_prices: dict, modelled_year: int) -> float:
    """Get ROC buyout price for a given year, interpolating if needed."""
    if not buyout_prices:
        return 0.0
    prices = {int(k): v for k, v in buyout_prices.items()}
    if modelled_year in prices:
        return prices[modelled_year]
    years = sorted(prices.keys())
    if modelled_year <= years[0]:
        return prices[years[0]]
    if modelled_year >= years[-1]:
        return prices[years[-1]]
    for i in range(len(years) - 1):
        if years[i] <= modelled_year <= years[i + 1]:
            frac = (modelled_year - years[i]) / (years[i + 1] - years[i])
            return prices[years[i]] + frac * (prices[years[i + 1]] - prices[years[i]])
    return 0.0


def _build_renewable_mc_lookup(renewable_mcs, network):
    """Match renewable empirical MC keys to network generator names.

    Strategy: direct name match, then substring match (both directions).
    Only matches generators with a renewable carrier.
    """
    if not renewable_mcs:
        return {}

    lookup = {}
    sorted_stations = sorted(renewable_mcs.keys(), key=len, reverse=True)

    for gen_name in network.generators.index:
        carrier = network.generators.at[gen_name, 'carrier']
        if carrier not in RENEWABLE_CARRIERS:
            continue

        # Direct match
        if gen_name in renewable_mcs:
            lookup[gen_name] = renewable_mcs[gen_name]
            continue

        # Substring match (both directions)
        base = gen_name.lower().split('__agg')[0].strip()
        base_ns = base.replace(' ', '').replace('_', '')
        for st in sorted_stations:
            st_ns = st.lower().replace(' ', '').replace('_', '')
            if st_ns in base_ns or base_ns in st_ns:
                lookup[gen_name] = renewable_mcs[st]
                break

    logger.info(f"Renewable MC lookup: {len(lookup)} matched "
                f"(from {len(renewable_mcs)} calibrated stations)")
    return lookup


def _compute_carrier_factors(thermal_daily_df, fuel_prices, carbon_price,
                             scenario_config=None):
    """Derive carrier correction factors from empirical data or external file.

    When carrier_correction_factors.enabled=True and a file is configured,
    loads pre-computed factors (e.g. from historical ELEXON calibration).
    Otherwise, computes inline from thermal daily empirical data (historical behaviour).

    Returns {carrier: factor} where factor = median(empirical) / formula.
    """
    mc_config = (scenario_config or {}).get('marginal_costs', {})
    ccf_config = mc_config.get('carrier_correction_factors', {})

    # Historical ELEXON correction factors are calibration aids, not future
    # fuel-cost projections. A future scenario must explicitly opt in to use
    # them; the WP3 default is formula costs from its commodity assumptions.
    modelled_year = (scenario_config or {}).get('modelled_year')
    if (
        modelled_year and modelled_year > 2024
        and ccf_config.get('enabled', False)
        and not ccf_config.get('allow_future_historical_factors', False)
    ):
        logger.info(
            "Skipping historical carrier correction factors for future scenario; "
            "set allow_future_historical_factors: true to override"
        )
        return {}

    # ── External file mode (for future scenarios) ─────────────────────────────
    if ccf_config.get('enabled', False) and ccf_config.get('file'):
        return _load_carrier_factors_from_file(ccf_config, scenario_config)

    # ── Inline mode (for historical scenarios with empirical data) ────────────
    if thermal_daily_df is None or thermal_daily_df.empty:
        return {}

    CARRIER_EFF = FORMULA_CARRIER_EFF  # FIX-2

    carrier_medians = thermal_daily_df.groupby('carrier')['empirical_mc'].median()
    factors = {}
    for carrier, emp_mc in carrier_medians.items():
        eff = CARRIER_EFF.get(carrier)
        if eff is None:
            continue
        fp = fuel_prices.get(carrier, 0)
        _, _, formula_mc = calculate_marginal_cost(carrier, fp, carbon_price, eff)
        if formula_mc > 0:
            factors[carrier] = emp_mc / formula_mc
            logger.info(f"  Carrier factor {carrier}: {factors[carrier]:.3f} "
                        f"(empirical {emp_mc:.1f} / formula {formula_mc:.1f})")
    return factors


def _load_carrier_factors_from_file(ccf_config, scenario_config):
    """Load carrier correction factors from external CSV.

    Selection logic:
      - source_year="closest": pick year nearest to modelled_year
      - source_year="median": use aggregated median rows
      - source_year=<int>: use that specific year
    """
    # Prefer Snakemake-resolved path, fall back to config value
    filepath = Path(
        scenario_config.get('_correction_factors_file') or ccf_config['file']
    )
    if not filepath.exists():
        logger.warning(f"Carrier correction factors file not found: {filepath}")
        return {}

    fallback = ccf_config.get('fallback', 1.0)
    source_year = ccf_config.get('source_year', 'closest')
    modelled_year = (scenario_config or {}).get('modelled_year', 2035)

    try:
        df = pd.read_csv(filepath)
        if not {'source_year', 'carrier', 'correction_factor'}.issubset(df.columns):
            logger.warning(f"Carrier correction factors CSV missing required columns")
            return {}

        # Select rows based on source_year config
        if source_year == 'median':
            selected = df[df['source_year'] == 'median']
        elif source_year == 'closest':
            numeric_years = df[df['source_year'] != 'median']['source_year'].astype(int)
            if len(numeric_years) == 0:
                selected = df[df['source_year'] == 'median']
            else:
                closest_year = int(numeric_years.iloc[
                    (numeric_years - modelled_year).abs().argmin()
                ])
                selected = df[df['source_year'].astype(str) == str(closest_year)]
                logger.info(f"  Using correction factors from closest year: {closest_year}")
        else:
            # Specific year
            selected = df[df['source_year'].astype(str) == str(source_year)]
            if selected.empty:
                logger.warning(f"No correction factors for source_year={source_year}, "
                               f"falling back to median")
                selected = df[df['source_year'] == 'median']

        if selected.empty:
            logger.warning("No matching correction factors found")
            return {}

        factors = {}
        for _, row in selected.iterrows():
            carrier = row['carrier']
            factor = float(row['correction_factor'])
            # Skip nuclear — nuclear_override takes precedence
            if carrier in ('nuclear', 'Nuclear'):
                logger.info(f"  Skipping nuclear correction factor ({factor:.3f}) — "
                            f"nuclear_override takes precedence")
                continue
            factors[carrier] = factor
            logger.info(f"  Loaded carrier factor {carrier}: {factor:.3f} "
                        f"(from file, source_year={row['source_year']})")

        return factors

    except Exception as e:
        logger.warning(f"Failed to load carrier correction factors: {e}")
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN: APPLY MARGINAL COSTS
# ══════════════════════════════════════════════════════════════════════════════

def _apply_dpa_dispatch(network, mc_df: pd.DataFrame, sub_config: dict) -> pd.DataFrame:
    """Dispatch Power Agreement plant runs ahead of unabated gas.

    A DPA's variable payment is set so the CCS plant is dispatched ahead of
    unabated gas generation (DESNZ, DPA terms): its dispatch cost is its own
    marginal cost or the cheapest unabated CCGT's less ``dpa_margin_gbp_per_mwh``,
    whichever is lower. Its payments are settled on bills outside the CfD line.
    """
    if "support_type" not in network.generators.columns or mc_df.empty:
        return mc_df
    dpa = network.generators.index[network.generators["support_type"].astype(str).str.strip().eq("DPA")]
    if not len(dpa):
        return mc_df
    ccgt = mc_df[mc_df["carrier"].eq("CCGT")]
    if ccgt.empty:
        logger.warning("DPA plant found but no unabated CCGT to rank it against; keeping its own marginal cost")
        return mc_df
    margin = float(sub_config.get("dpa_margin_gbp_per_mwh", 1.0))
    reference = float(ccgt["marginal_cost"].min()) - margin
    rows = mc_df["generator"].isin(dpa)
    own = mc_df.loc[rows, "marginal_cost"].astype(float)
    mc_df.loc[rows, "marginal_cost"] = own.clip(upper=reference)
    mc_df.loc[rows, "source"] = mc_df.loc[rows, "source"].astype(str) + f"|dpa_ahead_of_unabated({reference:.2f})"
    network.generators.loc[mc_df.loc[rows, "generator"], "marginal_cost"] = mc_df.loc[rows, "marginal_cost"].to_numpy()
    logger.info(f"DPA dispatch: {len(dpa)} unit(s) at or below {reference:.2f} GBP/MWh "
                f"(cheapest unabated CCGT less {margin:.2f})")
    return mc_df


def apply_marginal_costs_to_network(network, fuel_prices, carbon_price,
                                    scenario_config=None):
    """Apply marginal costs to all generators using flat two-tier logic.

    Priority for each generator:
      1. Protected carriers (load_shedding) -> skip
      2. Nuclear -> config.nuclear_mc
      3. Thermal + in thermal_empirical_mc -> daily empirical value
      4. Thermal + not in empirical -> formula x carrier_factor
      5. Renewable + in renewable_empirical_mc -> empirical value
      6. Renewable + subsidy fallback -> CfD=0, ROC=negative, merchant=0
      7. Everything else -> formula MC
    """
    logger.info("Applying marginal costs to generators")
    logger.info(f"Total generators: {len(network.generators)}")

    mc_config = (scenario_config or {}).get('marginal_costs', {})
    modelled_year = (scenario_config or {}).get('modelled_year', 2024)
    sub_config = (scenario_config or {}).get('support', {}).get('dispatch_costs', {})
    future_assumptions = get_future_cost_assumptions(scenario_config)
    future_case_enabled = bool(future_assumptions)
    future_variable_om = future_assumptions.get('variable_om', {})
    uk_ets, carbon_price_support = _future_carbon_components(future_assumptions)
    emission_factors = carbon_emission_factors(future_assumptions)
    if emission_factors is not CARBON_EMISSION_FACTORS:
        logger.info(f"Gas carbon on the net-CV factor: {emission_factors['CCGT']:.1f} kgCO2/MWh_th "
                    f"(CCS Gas {emission_factors['CCS Gas']:.2f})")

    # Nuclear MC
    nuclear_mc = mc_config.get('nuclear_mc', 10.0)

    # Load thermal empirical MCs
    thermal_mc_file = (scenario_config or {}).get('_thermal_mc_file')
    thermal_daily_df = _load_thermal_empirical_mc(thermal_mc_file)

    # Load renewable empirical MCs
    renewable_mc_file = (scenario_config or {}).get('_renewable_mc_file')
    renewable_mcs_raw = _load_renewable_empirical_mc(renewable_mc_file)
    renewable_mc_lookup = _build_renewable_mc_lookup(renewable_mcs_raw, network)

    # Compute carrier correction factors inline from thermal daily data
    carrier_factors = _compute_carrier_factors(thermal_daily_df, fuel_prices, carbon_price,
                                               scenario_config=scenario_config)

    # Build thermal empirical lookup: {generator: median_mc}
    thermal_mc_lookup = {}
    if thermal_daily_df is not None and not thermal_daily_df.empty:
        thermal_mc_lookup = (
            thermal_daily_df.groupby('generator')['empirical_mc']
            .median().to_dict()
        )
        logger.info(f"Thermal empirical MC lookup: {len(thermal_mc_lookup)} generators")

    # Subsidy config
    subsidy_enabled = sub_config.get('enabled', False)
    cfd_dispatch_cost = sub_config.get('cfd_dispatch_cost', 0.0)
    cfd_dispatch_exempt = set(sub_config.get('cfd_dispatch_cost_exempt_carriers', []) or [])
    # Item MR5: plant whose fuel is paid for by another product. Energy from
    # waste is paid a gate fee to burn waste it must burn anyway; gas CHP sells
    # heat that would otherwise come from a boiler.
    byproduct_adjustments = mc_config.get('byproduct_adjustments', {}) or {}
    roc_buyout = 0.0
    if subsidy_enabled:
        buyout_prices = sub_config.get('roc_buyout_prices', {})
        roc_buyout = _get_roc_buyout_price(buyout_prices, modelled_year)
        logger.info(f"Subsidies enabled: CfD={cfd_dispatch_cost}, ROC buyout={roc_buyout:.2f}")

    # Biofuel subsidy config — deducts ROC income from thermal MC for biomass/biogas
    # carriers. Separate from the renewable subsidy toggle because biomass is thermal.
    biofuel_cfg = mc_config.get('biofuel_subsidy', {})
    biofuel_subsidy_enabled = biofuel_cfg.get('enabled', False)
    biofuel_roc_buyout = 0.0
    biofuel_default_bandings = {}
    biofuel_mc_floor = biofuel_cfg.get('mc_floor', 5.0)  # Prevent negative MC from ROC income
    biofuel_require_support_label = biofuel_cfg.get('require_support_label', False)
    # A biofuel unit explicitly labelled ROC (Ofgem register or proxy share)
    # forgoes banding x buyout when it does not run, exactly as ROC wind and
    # solar do. With the renewable subsidy layer on, it takes the same ROC
    # value even when the carrier-wide biofuel deduction is off (CP30: 830 MW
    # of ROC biomass sat at the 128.57 formula cost while ROC wind bid
    # -(banding x buyout); audit 27 Sep 2026). Must match config/defaults/base/marginal_costs.yaml.
    roc_labelled_follow_subsidies = bool(biofuel_cfg.get('roc_labelled_follow_subsidies', True))
    if biofuel_subsidy_enabled:
        # Use explicit buyout price if given, else fall back to the renewable ROC table
        if 'roc_buyout_price' in biofuel_cfg:
            biofuel_roc_buyout = float(biofuel_cfg['roc_buyout_price'])
        else:
            buyout_prices = sub_config.get('roc_buyout_prices', {})
            biofuel_roc_buyout = _get_roc_buyout_price(buyout_prices, modelled_year)
        biofuel_default_bandings = biofuel_cfg.get('default_roc_bandings', {})
        logger.info(f"Biofuel subsidy enabled: ROC buyout={biofuel_roc_buyout:.2f}, "
                    f"bandings={biofuel_default_bandings}")

    cp30_mc_overrides = get_cp30_marginal_cost_overrides(scenario_config)
    if cp30_mc_overrides:
        logger.info(f"CP30 marginal-cost overrides enabled: {cp30_mc_overrides}")

    # Normalize unclassified carriers
    if 'carrier' in network.generators.columns:
        mask = network.generators['carrier'] == 'unclassified'
        if mask.any():
            logger.warning(f"Normalizing {mask.sum()} 'unclassified' generators to 'OCGT'")
            network.generators.loc[mask, 'carrier'] = 'OCGT'

    # ── Main loop ────────────────────────────────────────────────────────────
    mc_data = []

    for gen_name, gen in network.generators.iterrows():
        carrier = gen.get('carrier', 'unknown')
        efficiency = gen.get('efficiency', 1.0)
        p_nom = gen.get('p_nom', 0)
        existing_mc = gen.get('marginal_cost', 0.0)
        efficiency_used = efficiency
        fuel_component = 0.0
        carbon_component = 0.0
        variable_om = 0.0
        heat_credit = np.nan
        mc_before_heat_credit = np.nan

        # 1. Protected — never touch
        if carrier in PROTECTED_CARRIERS:
            mc_data.append({'generator': gen_name, 'carrier': carrier,
                            'p_nom_MW': p_nom, 'marginal_cost': existing_mc,
                            'source': 'protected',
                            'efficiency_used': efficiency_used,
                            'fuel_cost_gbp_per_mwh': fuel_component,
                            'carbon_cost_gbp_per_mwh': carbon_component,
                            'variable_om_gbp_per_mwh': variable_om,
                            'future_cost_case': future_assumptions.get('case'),
                            'price_year': future_assumptions.get('price_year')})
            continue

        # 2. Nuclear — fixed override
        if carrier in ('nuclear', 'Nuclear', 'PWR', 'AGR'):
            mc = nuclear_mc
            source = 'nuclear_override'
        # 3. CP30 clean dispatchable assumptions
        elif carrier in cp30_mc_overrides and not future_case_enabled:
            mc = cp30_mc_overrides[carrier]
            source = 'cp30_assumption'

        # CfD support is a unit-level dispatch condition and takes precedence
        # over empirical or carrier-level thermal costs.
        elif (
            carrier in THERMAL_CARRIERS
            and subsidy_enabled
            and carrier not in cfd_dispatch_exempt
            and str(gen.get('support_type', '') or '').strip() == 'CfD'
        ):
            mc = cfd_dispatch_cost
            source = 'subsidy_CfD_dispatchable'

        # 4. Thermal with empirical MC
        elif (
            gen_name in thermal_mc_lookup
            and str(gen.get('support_type', '') or '').strip() != 'ROC'
        ):
            mc = thermal_mc_lookup[gen_name]
            source = 'thermal_empirical'

        # 4. Thermal without empirical — formula x carrier factor
        elif carrier in THERMAL_CARRIERS:
            fp = fuel_prices.get(carrier, 0)
            efficiency_used = resolve_thermal_efficiency(
                gen_name, gen, scenario_config
            )
            fuel_component, carbon_component, formula_mc = calculate_marginal_cost(
                carrier, fp, carbon_price, efficiency_used, emission_factors
            )
            factor = carrier_factors.get(carrier, 1.0)
            variable_om = _lookup_carrier_value(future_variable_om, carrier)
            mc = formula_mc * factor + variable_om
            source = f'formula*{factor:.3f}' if factor != 1.0 else 'formula'
            if variable_om:
                source = f'{source}+vom'

            adjustment = byproduct_adjustments.get(carrier)
            if adjustment:
                mode = adjustment.get('mode')
                if mode == 'gate_fee_covers_fuel_and_carbon':
                    # The waste is burned under disposal contracts whether or not
                    # the power is worth much; its fuel and carbon are paid from
                    # the gate fee, so the electricity's own cost is VOM only.
                    fuel_component, carbon_component = 0.0, 0.0
                    mc = variable_om
                    source = f'{source}:gate_fee_covers_fuel_and_carbon'
                elif mode == 'heat_credit':
                    heat_fuel = fuel_prices.get(adjustment.get('heat_fuel', 'gas'), 0.0)
                    # The boiler's gas on the unit's own basis, so the credit's
                    # carbon is the unit's carbon per MWh of fuel (the split in
                    # nationalised_institutions.heat_credit_rates relies on it).
                    heat_ef = emission_factors.get(adjustment.get('heat_fuel', 'gas'), 0.0)
                    boiler = float(adjustment['boiler_efficiency'])
                    credit = float(adjustment['heat_to_power']) * (
                        heat_fuel + heat_ef / 1000.0 * carbon_price) / boiler
                    heat_credit, mc_before_heat_credit = credit, mc
                    mc = max(mc - credit, 0.0)
                    source = f'{source}-heat_credit({credit:.1f})'
                else:
                    raise ValueError(f"Unknown byproduct adjustment mode for {carrier}: {mode!r}")

            # 4b. Biofuel ROC subsidy deduction — biomass/biogas plants receive
            # ROC income per MWh that reduces their effective wholesale MC.
            support_type = str(gen.get('support_type', '') or '').strip()
            labelled_roc = support_type == 'ROC'
            legacy_unlabelled = not biofuel_require_support_label and not support_type
            carrier_wide = biofuel_subsidy_enabled and (labelled_roc or legacy_unlabelled)
            roc_label_only = (
                not carrier_wide and labelled_roc and subsidy_enabled
                and roc_labelled_follow_subsidies
            )
            if carrier in BIOFUEL_CARRIERS and (carrier_wide or roc_label_only):
                buyout_used = biofuel_roc_buyout if carrier_wide else roc_buyout
                ro_banding = gen.get('ro_banding', None)
                if ro_banding is None or pd.isna(ro_banding) or ro_banding == 0:
                    ro_banding = (biofuel_default_bandings or biofuel_cfg.get('default_roc_bandings', {}) or {}).get(carrier, 0.0)
                if ro_banding and buyout_used > 0:
                    subsidy_income = float(ro_banding) * buyout_used
                    mc = mc - subsidy_income
                    # Floor: prevent ROC income making biomass cost-free/negative to the system
                    # Negative MC causes unconditional 100% dispatch which is unrealistic
                    mc = max(mc, biofuel_mc_floor)
                    # The "_ROC(" marker lets the resource-cost basis
                    # (scripts/market/marginal_cost_basis.py) strip this layer.
                    source = f'{source}_ROC({float(ro_banding):.4g}x{buyout_used:.2f})'

        # 5. Renewable — subsidy type takes priority, then empirical fallback
        #    CfD generators always get cfd_dispatch_cost (typically £0) because
        #    their ELEXON BM bid prices reflect CfD opportunity cost, not SRMC.
        #    ROC generators always get negative MC from banding × buyout.
        #    Empirical MC only applies to merchant/unclassified renewables.
        elif carrier in RENEWABLE_CARRIERS:
            support_type = gen.get('support_type', None)
            if subsidy_enabled and support_type == 'CfD':
                mc = cfd_dispatch_cost
                source = 'subsidy_CfD'
            elif subsidy_enabled and support_type == 'ROC':
                ro_banding = gen.get('ro_banding', 0.0)
                if pd.notna(ro_banding) and ro_banding > 0:
                    mc = -(float(ro_banding) * roc_buyout)
                    source = 'subsidy_ROC'
                else:
                    mc = 0.0
                    source = 'renewable_default'
            elif gen_name in renewable_mc_lookup:
                mc = renewable_mc_lookup[gen_name]
                source = 'renewable_empirical'
            else:
                mc = 0.0
                source = 'renewable_default'

        # 7. Everything else — formula
        else:
            fp = fuel_prices.get(carrier, 0)
            fuel_component, carbon_component, mc = calculate_marginal_cost(
                carrier, fp, carbon_price, efficiency, emission_factors
            )
            source = 'formula'

        network.generators.loc[gen_name, 'marginal_cost'] = mc
        mc_data.append({'generator': gen_name, 'carrier': carrier,
                        'p_nom_MW': p_nom, 'marginal_cost': mc,
                        'source': source,
                        'efficiency_used': efficiency_used,
                        'emission_factor_kgco2_per_mwh_th':
                            emission_factors.get(carrier, 0.0),
                        'fuel_cost_gbp_per_mwh': fuel_component,
                        'carbon_cost_gbp_per_mwh': carbon_component,
                        'variable_om_gbp_per_mwh': variable_om,
                        'heat_credit_gbp_per_mwh': heat_credit,
                        'mc_before_heat_credit_gbp_per_mwh': mc_before_heat_credit,
                        'carbon_uk_ets_gbp_per_tco2': uk_ets if future_case_enabled else np.nan,
                        'carbon_price_support_gbp_per_tco2': carbon_price_support if future_case_enabled else np.nan,
                        'future_cost_case': future_assumptions.get('case'),
                        'price_year': future_assumptions.get('price_year')})

    mc_df = pd.DataFrame(mc_data)
    mc_df = _apply_dpa_dispatch(network, mc_df, sub_config)

    # ── Logging ──────────────────────────────────────────────────────────────
    active = mc_df[mc_df['source'] != 'protected']
    protected = mc_df[mc_df['source'] == 'protected']
    if len(protected):
        logger.info(f"Protected {len(protected)} generators (load_shedding etc.)")

    logger.info("\nMarginal cost summary:")
    logger.info("=" * 80)
    if len(active) > 0:
        summary = active.groupby('carrier').agg(
            count=('generator', 'count'),
            capacity=('p_nom_MW', 'sum'),
            mean_mc=('marginal_cost', 'mean'),
            min_mc=('marginal_cost', 'min'),
            max_mc=('marginal_cost', 'max'),
        ).round(2)
        for carrier in summary.index:
            r = summary.loc[carrier]
            logger.info(f"  {carrier:20s}: {int(r['count']):4d} units, {r['capacity']:8.1f} MW, "
                        f"MC: {r['mean_mc']:6.2f} ({r['min_mc']:.2f}-{r['max_mc']:.2f})")
    logger.info("=" * 80)

    # Source breakdown
    source_counts = active.groupby('source').size()
    logger.info("MC source breakdown:")
    for src, count in source_counts.items():
        logger.info(f"  {src}: {count} generators")

    return mc_df


# ══════════════════════════════════════════════════════════════════════════════
# SEASONAL HEAT CREDIT AND MUST-RUN FLOOR (heat-led CHP)
# ══════════════════════════════════════════════════════════════════════════════
#
# The heat credit (byproduct_adjustments heat_credit) and the CHP must-run floor
# (thermal.availability_factors.min_output) were flat all year, so gas CHP ran at
# a 66 GBP/MWh heat-credited cost with a 30% floor in July as in January: CF 0.58
# against CP30's 0.36, and its busiest month was July. Both now follow heat
# demand. User decision, 26 Sep 2026.
#
# Shape: heat demand h(m) = base + (1 - base) x HDD(m) / max HDD, with ``base``
# the process-heat share that does not follow the weather (industry holds most
# GB CHP capacity; DUKES chapter 7) and HDD the monthly heating degree days.
# The default HDD are the 2010-2024 means over GB land cells of this project's
# ERA5 cutouts (resources/atlite/cutouts/uk-{year}.nc, cells with height > 1 m,
# base 15.5 C, the base build_heat_profiles.py uses): 2,052 degree days a year,
# 34 in July against 306 in January. h is then scaled to an hour-weighted
# calendar-year mean of 1, so the annual heat credit and floor are the configured
# ones and only their timing moves: with base 0.5 the credit and floor run from
# 0.71x (August) to 1.28x (January) of their annual value.
DEFAULT_MONTHLY_HEATING_DEGREE_DAYS = (
    306.3, 273.2, 277.6, 219.4, 148.7, 72.0, 34.0, 32.1, 64.3, 132.8, 216.8, 274.8,
)
DEFAULT_PROCESS_HEAT_SHARE = 0.5


def seasonal_heat_factors(snapshots: pd.DatetimeIndex, seasonal_cfg: dict | None) -> pd.Series:
    """Heat-demand factor per snapshot, mean 1 over the snapshots' calendar year."""
    cfg = seasonal_cfg or {}
    hdd = np.asarray(cfg.get('monthly_heating_degree_days') or DEFAULT_MONTHLY_HEATING_DEGREE_DAYS, dtype=float)
    if hdd.shape != (12,) or (hdd < 0).any() or hdd.max() <= 0:
        raise ValueError("seasonal.monthly_heating_degree_days must be 12 non-negative values, not all zero")
    base = float(cfg.get('base_load_share', DEFAULT_PROCESS_HEAT_SHARE))
    if not 0.0 <= base <= 1.0:
        raise ValueError(f"seasonal.base_load_share must be in [0, 1], got {base}")
    monthly = base + (1.0 - base) * hdd / hdd.max()
    snapshots = pd.DatetimeIndex(snapshots)
    year = int(snapshots[0].year) if len(snapshots) else 2030
    days = np.array([pd.Period(f"{year}-{m:02d}").days_in_month for m in range(1, 13)], dtype=float)
    monthly = monthly / float((monthly * days).sum() / days.sum())
    return pd.Series(monthly[snapshots.month - 1], index=snapshots, name='heat_factor')


def apply_seasonal_heat_profile(network, mc_df: pd.DataFrame, scenario_config: dict | None) -> pd.DataFrame:
    """Make heat-credited generators' marginal cost and must-run floor follow heat demand.

    For each carrier whose ``byproduct_adjustments`` entry is ``heat_credit`` with
    ``seasonal.enabled`` (default true): ``generators_t.marginal_cost`` becomes
    ``max(mc_before_credit - credit x h(t), 0)`` and, with ``apply_to_min_output``
    (default true), ``generators_t.p_min_pu`` becomes ``p_min_pu x h(t)`` capped
    at ``p_max_pu``. The static ``marginal_cost`` is set to the snapshot mean so
    static readers see the annual value. Returns ``mc_df`` with the mean written
    back and the seasonal range recorded.
    """
    mc_config = (scenario_config or {}).get('marginal_costs', {}) or {}
    adjustments = mc_config.get('byproduct_adjustments', {}) or {}
    if 'heat_credit_gbp_per_mwh' not in mc_df.columns or len(network.snapshots) == 0:
        return mc_df
    snapshots = network.snapshots
    for carrier, adjustment in adjustments.items():
        if (adjustment or {}).get('mode') != 'heat_credit':
            continue
        seasonal = adjustment.get('seasonal', {}) or {}
        if not bool(seasonal.get('enabled', True)):
            logger.info(f"Seasonal heat profile: off for {carrier} (flat heat credit and floor)")
            continue
        rows = mc_df[(mc_df['carrier'] == carrier) & mc_df['heat_credit_gbp_per_mwh'].notna()]
        rows = rows[rows['generator'].isin(network.generators.index)]
        if rows.empty:
            continue
        factor = seasonal_heat_factors(snapshots, seasonal)
        names = rows['generator'].tolist()
        credit = rows['heat_credit_gbp_per_mwh'].to_numpy(dtype=float)
        before = rows['mc_before_heat_credit_gbp_per_mwh'].to_numpy(dtype=float)
        mc_t = pd.DataFrame(
            np.maximum(before[None, :] - credit[None, :] * factor.to_numpy()[:, None], 0.0),
            index=snapshots, columns=names,
        )
        varying = network.generators_t.marginal_cost
        if varying is None or varying.empty:
            varying = pd.DataFrame(index=snapshots)
        varying = varying.reindex(index=snapshots)
        network.generators_t['marginal_cost'] = pd.concat(
            [varying.drop(columns=[n for n in names if n in varying.columns]), mc_t], axis=1
        )
        mean_mc = mc_t.mean()
        network.generators.loc[names, 'marginal_cost'] = mean_mc.to_numpy()
        index = rows.index
        mc_df.loc[index, 'marginal_cost'] = mean_mc.to_numpy()
        mc_df.loc[index, 'marginal_cost_seasonal_min'] = mc_t.min().to_numpy()
        mc_df.loc[index, 'marginal_cost_seasonal_max'] = mc_t.max().to_numpy()
        mc_df.loc[index, 'source'] = mc_df.loc[index, 'source'].astype(str) + '|seasonal_heat'

        floor_note = ''
        if bool(seasonal.get('apply_to_min_output', True)):
            static_min = pd.to_numeric(network.generators.loc[names, 'p_min_pu'], errors='coerce').fillna(0.0)
            active = static_min[static_min > 0].index.tolist()
            if active:
                static_max = pd.to_numeric(network.generators.loc[active, 'p_max_pu'], errors='coerce').fillna(1.0)
                cap = pd.DataFrame(np.tile(static_max.to_numpy(), (len(snapshots), 1)),
                                   index=snapshots, columns=active)
                max_t = network.generators_t.p_max_pu
                if max_t is not None and not max_t.empty:
                    cols = [n for n in active if n in max_t.columns]
                    if cols:
                        cap[cols] = max_t[cols].reindex(snapshots).fillna(static_max[cols]).to_numpy()
                floor_t = pd.DataFrame(
                    static_min[active].to_numpy()[None, :] * factor.to_numpy()[:, None],
                    index=snapshots, columns=active,
                ).clip(lower=0.0)
                floor_t = floor_t.where(floor_t <= cap, cap)
                min_t = network.generators_t.p_min_pu
                if min_t is None or min_t.empty:
                    min_t = pd.DataFrame(index=snapshots)
                min_t = min_t.reindex(index=snapshots)
                network.generators_t['p_min_pu'] = pd.concat(
                    [min_t.drop(columns=[n for n in active if n in min_t.columns]), floor_t], axis=1
                )
                floor_note = (f"; must-run floor {float(static_min[active].mean()):.2f} -> "
                              f"{float(floor_t.min().mean()):.2f}-{float(floor_t.max().mean()):.2f} "
                              f"({len(active)} units)")
        by_month = mc_t.mean(axis=1).groupby(snapshots.month).mean()
        logger.info(
            f"Seasonal heat profile for {carrier}: {len(names)} units, heat factor "
            f"{factor.min():.2f}-{factor.max():.2f} (mean 1 over the year); marginal cost "
            f"{by_month.min():.2f} (month {int(by_month.idxmin())}) to {by_month.max():.2f} "
            f"(month {int(by_month.idxmax())}), annual mean {float(mean_mc.mean()):.2f} GBP/MWh"
            + floor_note
        )
    return mc_df


# ══════════════════════════════════════════════════════════════════════════════
# TIME-VARYING MC (daily empirical)
# ══════════════════════════════════════════════════════════════════════════════

def apply_time_varying_mc(network, scenario_config, daily_mc_file=None):
    """Convert static MCs to hourly time series using daily empirical MCs."""
    mc_cfg = scenario_config.get('marginal_costs', {})
    emp_cfg = mc_cfg.get('empirical_calibration', {})

    daily_enabled = emp_cfg.get('enabled', False) and daily_mc_file

    if not daily_enabled:
        logger.info("Time-varying MC: disabled")
        return

    logger.info("=" * 60)
    logger.info("APPLYING TIME-VARYING MARGINAL COSTS")
    logger.info("=" * 60)

    snapshots = network.snapshots
    gen_names = network.generators.index
    static_mc = network.generators['marginal_cost']

    # Start with static MC broadcast
    mc_t = pd.DataFrame(
        np.tile(static_mc.values, (len(snapshots), 1)),
        index=snapshots, columns=gen_names, dtype=float,
    )

    # ── Daily empirical MC override ──────────────────────────────────────────
    # Nuclear carriers must be excluded: their ELEXON bids reflect strategic
    # BM behaviour (high bids to avoid turn-down), not short-run marginal cost.
    # The nuclear_override (Layer 2) must take precedence.
    nuclear_carriers = {'nuclear', 'Nuclear', 'PWR', 'AGR'}
    gen_carriers = network.generators['carrier']
    skip_nuclear = set(gen_carriers[gen_carriers.isin(nuclear_carriers)].index)

    try:
        daily_df = pd.read_csv(daily_mc_file)
        if 'date' in daily_df.columns:
            daily_df['date'] = pd.to_datetime(daily_df['date']).dt.date
            daily_lookup = daily_df.set_index(['generator', 'date'])['empirical_mc']
            snap_dates = pd.Series(snapshots).dt.date.values

            overridden = 0
            skipped_nuclear = 0
            for gen in gen_names:
                if gen in skip_nuclear:
                    skipped_nuclear += 1
                    continue
                for d in sorted(set(snap_dates)):
                    if (gen, d) in daily_lookup.index:
                        emp_mc = daily_lookup.loc[(gen, d)]
                        if pd.notna(emp_mc) and emp_mc > 0:
                            mask = snap_dates == d
                            mc_t.loc[mask, gen] = emp_mc
                            overridden += 1

            logger.info(f"  Daily MC: overrode {overridden} (generator, date) cells")
            if skipped_nuclear:
                logger.info(f"  Preserved nuclear_override for {skipped_nuclear} nuclear generators")
    except Exception as e:
        logger.warning(f"  Failed to load daily MCs: {e}")

    # ── Carrier MC adjustments ───────────────────────────────────────────────
    # Additive corrections per carrier applied AFTER empirical calibration.
    # Primary use: widen coal-gas gap that empirical switch-on methodology
    # compresses (both fuels turn on at similar wholesale prices, but coal has
    # higher CO₂ costs that the switch-on method does not capture).
    adjustments = emp_cfg.get('carrier_mc_adjustments', {})
    if adjustments:
        gen_carriers = network.generators['carrier']
        adj_count = 0
        for carrier, adj_value in adjustments.items():
            adj_value = float(adj_value)
            carrier_mask = gen_carriers == carrier
            if not carrier_mask.any():
                continue
            carrier_gens = gen_carriers[carrier_mask].index
            mc_t[carrier_gens] = mc_t[carrier_gens] + adj_value
            adj_count += len(carrier_gens)
            logger.info(
                f"  MC adjustment: {carrier} += {adj_value:+.1f} £/MWh "
                f"({len(carrier_gens)} generators)"
            )
        if adj_count:
            logger.info(f"  Applied MC adjustments to {adj_count} generators")

    network.generators_t['marginal_cost'] = mc_t
    logger.info(f"  Result: {mc_t.shape[0]} snapshots x {mc_t.shape[1]} generators")
    logger.info(f"  MC range: {mc_t.min().min():.2f}-{mc_t.max().max():.2f}/MWh")


# ══════════════════════════════════════════════════════════════════════════════
# SNAKEMAKE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Snakemake entry point."""
    global logger

    log_path = snakemake.log[0] if hasattr(snakemake, 'log') and snakemake.log else "apply_marginal_costs"
    logger = setup_logging(log_path)

    logger.info("=" * 80)
    logger.info("APPLYING MARGINAL COSTS TO NETWORK")
    logger.info("=" * 80)

    try:
        network = load_network(snakemake.input.network, custom_logger=logger)
        scenario_config = snakemake.params.scenario_config

        # Pass file paths through scenario_config for the apply function
        if hasattr(snakemake.params, 'fuel_price_file') and snakemake.params.fuel_price_file:
            scenario_config['_fuel_price_file'] = snakemake.params.fuel_price_file
        if hasattr(snakemake.params, 'carbon_price_file') and snakemake.params.carbon_price_file:
            scenario_config['_carbon_price_file'] = snakemake.params.carbon_price_file
        if hasattr(snakemake.params, 'thermal_mc_file') and snakemake.params.thermal_mc_file:
            scenario_config['_thermal_mc_file'] = snakemake.params.thermal_mc_file
        if hasattr(snakemake.params, 'renewable_mc_file') and snakemake.params.renewable_mc_file:
            scenario_config['_renewable_mc_file'] = snakemake.params.renewable_mc_file
        if hasattr(snakemake.params, 'correction_factors_file') and snakemake.params.correction_factors_file:
            scenario_config['_correction_factors_file'] = snakemake.params.correction_factors_file

        fuel_prices = load_fuel_prices(scenario_config)
        carbon_price = get_carbon_price(scenario_config)

        mc_df = apply_marginal_costs_to_network(
            network, fuel_prices, carbon_price, scenario_config
        )

        # Time-varying MC (daily empirical base)
        thermal_mc_file = getattr(snakemake.params, 'thermal_mc_file', None)
        apply_time_varying_mc(network, scenario_config, daily_mc_file=thermal_mc_file)

        # After the daily-empirical frame, which rebuilds generators_t.marginal_cost
        # from the static column and would otherwise flatten the seasonal credit.
        mc_df = apply_seasonal_heat_profile(network, mc_df, scenario_config)

        # Save outputs
        mc_df.to_csv(snakemake.output.marginal_costs_csv, index=False)
        save_network(network, snakemake.output.network, custom_logger=logger)

        logger.info("=" * 80)
        logger.info("MARGINAL COSTS APPLIED SUCCESSFULLY")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Failed to apply marginal costs: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
