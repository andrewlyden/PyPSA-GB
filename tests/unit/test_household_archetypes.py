"""Item G6: household archetypes, each with a standing and a unit split.

Decisions 11 and 12 (22 September 2026): Ofgem's low, medium and high typical
domestic consumption values plus heat-pump and electric-vehicle households; points,
not bands. The user retired 2,700 kWh for Ofgem's July 2026 medium, 2,500. A
single-rate tariff charges every household the same unit rate, so an archetype's
bill is the standing charge plus the unit rate times its consumption; what a heat
pump or an EV costs to serve is a separate diagnostic.

The standing and unit split replaces the old rule, which held every primary-source
line fixed and scaled everything else. That rule held per-kWh lines (the Capacity
Market, AAHEDC, the counterparty levies) fixed and let the public-ownership network
savings scale while the TNUoS and DUoS lines they reduce stayed fixed.
"""

import math

import pandas as pd
import pytest

from scripts.consumer_bill.household_archetypes import (
    archetype_bills, archetype_consumption, mixed_standing_share, standing_shares,
)

BLOCK = {
    "benchmark_kwh": 3100,
    "archetypes": [
        {"id": "ofgem_low", "kwh": 1600},
        {"id": "ofgem_medium", "kwh": 2500},
        {"id": "heat_pump", "base": "ofgem_medium",
         "added": {"gas_kwh": 9500, "boiler_efficiency": 0.85, "seasonal_performance_factor": 2.80}},
        {"id": "electric_vehicle", "base": "ofgem_medium", "added": {"miles": 9400, "kwh_per_mile": 0.30}},
    ],
}


def test_archetype_volumes_come_from_their_declared_inputs():
    kwh = archetype_consumption(BLOCK)
    assert kwh["ofgem_low"] == 1600 and kwh["ofgem_medium"] == 2500
    assert kwh["heat_pump"] == pytest.approx(2500 + 9500 * 0.85 / 2.80)
    assert kwh["electric_vehicle"] == pytest.approx(2500 + 9400 * 0.30)


def test_an_unknown_base_or_addition_is_refused():
    with pytest.raises(ValueError, match="base"):
        archetype_consumption({"archetypes": [{"id": "x", "base": "nope", "added": {"miles": 1, "kwh_per_mile": 1}}]})
    with pytest.raises(ValueError, match="added"):
        archetype_consumption({"archetypes": [{"id": "a", "kwh": 1}, {"id": "x", "base": "a", "added": {"lumens": 3}}]})


def test_the_mixed_share_reproduces_the_published_standing_charge():
    """The historical path's method: fixed lines fixed, mixed lines pro rata to hit the cap's nil."""
    calibration = {"nil_gbp_ex_vat": 198.8, "known_fixed_gbp": {"tnuos_residual": 91.3, "whd_residual": 3.4},
                   "mixed_lines_gbp": {"supplier": 165.2, "duos": 122.1, "smncc": 9.8}}
    share = mixed_standing_share(calibration)
    assert share == pytest.approx((198.8 - 91.3 - 3.4) / (165.2 + 122.1 + 9.8))
    with pytest.raises(ValueError, match="between 0 and 1"):
        mixed_standing_share({**calibration, "nil_gbp_ex_vat": 50.0})


def _components():
    return pd.DataFrame([
        {"component_id": "direct_energy", "annual_gbp_per_3100kwh_ex_vat": 300.0},
        {"component_id": "tnuos", "annual_gbp_per_3100kwh_ex_vat": 100.0},
        {"component_id": "supplier_costs_and_margin", "annual_gbp_per_3100kwh_ex_vat": 150.0},
        {"component_id": "tnuos_public_financing_adjustment", "annual_gbp_per_3100kwh_ex_vat": -10.0},
        {"component_id": "capacity_market", "annual_gbp_per_3100kwh_ex_vat": 10.0},
    ])


BASIS = {"standing_share": {"tnuos": 0.9, "supplier_costs_and_margin": 0.35},
         "inherit": {"tnuos_public_financing_adjustment": "tnuos"}}


def test_standing_shares_default_to_unit_and_savings_inherit_their_line():
    shares = standing_shares(_components(), BASIS)
    assert shares["direct_energy"] == 0.0 and shares["capacity_market"] == 0.0
    assert shares["tnuos"] == 0.9 and shares["tnuos_public_financing_adjustment"] == 0.9
    with pytest.raises(ValueError, match="between 0 and 1"):
        standing_shares(_components(), {"standing_share": {"tnuos": 1.5}})


def test_an_archetype_bill_is_the_standing_charge_plus_the_unit_rate_times_its_volume():
    bills = archetype_bills(_components(), standing_shares(_components(), BASIS),
                            {"benchmark": 3100.0, "ofgem_medium": 2500.0}, benchmark_kwh=3100, vat_rate=0.05)
    standing = 100 * 0.9 + 150 * 0.35 - 10 * 0.9
    total = 300 + 100 + 150 - 10 + 10
    assert bills.loc["benchmark", "ex_vat_gbp"] == pytest.approx(total)
    assert bills.loc["ofgem_medium", "standing_charge_gbp_ex_vat"] == pytest.approx(standing)
    assert bills.loc["ofgem_medium", "ex_vat_gbp"] == pytest.approx(standing + (total - standing) * 2500 / 3100)
    assert bills.loc["ofgem_medium", "unit_rate_p_per_kwh_ex_vat"] == pytest.approx((total - standing) / 3100 * 100)
    assert bills.loc["ofgem_medium", "total_gbp_inc_vat"] == pytest.approx(bills.loc["ofgem_medium", "ex_vat_gbp"] * 1.05)
    assert math.isclose(bills.loc["ofgem_medium", "vat_gbp"], bills.loc["ofgem_medium", "ex_vat_gbp"] * 0.05)
