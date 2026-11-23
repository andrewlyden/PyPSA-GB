"""Item OH1: what it costs the state to run the thing.

The overhead is the state's HOLDING COMPANY -- board, group finance and
treasury, legal, HR, IT, risk, regulatory and policy -- and not plant
operations.  DESNZ's fixed operating costs are the ones "a developer is liable
for ... associated with the generator and site itself", covering "staffing
costs and routine maintenance and any equipment contracts", so the
``fixed_om_gbp_per_kw_year`` column every case already carries is site-level and
the holding company is a genuine increment on top of it.

These tests pin the three things that would be easy to get wrong: the scope
(owned generation and storage, never the acquired networks, which keep their
RIIO allowed opex in the network account); the share (the state carries the
centre in proportion to what it owns); and the fact that the acquisition tariff
is built from components rather than from the required-revenue total, so the
overhead needs its own line there or it would be silently dropped.
"""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.public_ownership.calculate_financing_scenarios import calculate_financing_scenarios
from scripts.public_ownership.finance import (
    STATE_OVERHEAD_DEFAULT_RATES, annual_state_overhead, state_overhead_rate,
)
from scripts.public_ownership.levers import DEFAULT_SETTINGS, LEVER_STATE_OVERHEAD, LEVERS
from scripts.public_ownership.nationalised_tariff import LINE_STATE_OVERHEAD, NEW_LINE_LABELS
from tests.unit.test_public_ownership_finance import _costs
from tests.unit.test_public_ownership_levers import ROW
from config.config_loader import load_defaults


OVERHEAD = {"enabled": True, "central_gbp_per_kw_year": 3.0, "low_gbp_per_kw_year": 1.0,
            "high_gbp_per_kw_year": 8.0, "applies_to": ["Generator", "StorageUnit"]}


def _s(**levers):
    return dict(DEFAULT_SETTINGS, **levers)


# ── the rate ─────────────────────────────────────────────────────────────────


def test_the_lever_selects_the_configured_rate_and_the_twins_bracket_the_central():
    central = state_overhead_rate(OVERHEAD, _s())
    low = state_overhead_rate(OVERHEAD, _s(state_overhead="low"))
    high = state_overhead_rate(OVERHEAD, _s(state_overhead="high"))
    assert (low, central, high) == (1.0, 3.0, 8.0)
    assert low < central < high
    # A disabled block is the pre-OH1 behaviour: the overhead is not modelled.
    assert state_overhead_rate({"enabled": False, "central_gbp_per_kw_year": 3.0}, _s()) == 0.0
    with pytest.raises(ValueError, match="must not be negative"):
        state_overhead_rate({"central_gbp_per_kw_year": -1.0}, _s())


def test_the_code_default_matches_the_repository_config():
    """A missing config block must not silently move a number (house rule 6)."""
    block = load_defaults()["public_ownership"]
    configured = block["state_operating_overhead"]
    assert configured["enabled"] is True
    for value, key in (("central", "central_gbp_per_kw_year"), ("low", "low_gbp_per_kw_year"),
                       ("high", "high_gbp_per_kw_year")):
        assert configured[key] == STATE_OVERHEAD_DEFAULT_RATES[value], value
    assert configured["applies_to"] == ["Generator", "StorageUnit"]
    assert LEVERS[LEVER_STATE_OVERHEAD].default == "central"


# ── the scope ────────────────────────────────────────────────────────────────


def test_the_overhead_follows_the_state_share_and_only_the_components_in_scope():
    # 1 GW at 3 GBP/kW/yr is 3m GBP a year to a full owner, 0.9m to a 30% stake.
    assert annual_state_overhead(capacity_mw=1_000.0, ownership_share=1.0, rate_gbp_per_kw_year=3.0) == 3.0e6
    assert annual_state_overhead(capacity_mw=1_000.0, ownership_share=0.3, rate_gbp_per_kw_year=3.0) == pytest.approx(0.9e6)
    # Nothing owned, nothing to run.
    assert annual_state_overhead(capacity_mw=1_000.0, ownership_share=0.0, rate_gbp_per_kw_year=3.0) == 0.0
    # The acquired networks keep their RIIO allowed opex in the network finance
    # account, so an overhead on a network row would double-count.
    assert annual_state_overhead(
        capacity_mw=1_000.0, ownership_share=1.0, rate_gbp_per_kw_year=3.0,
        component_type="Line", applies_to=("Generator", "StorageUnit")) == 0.0
    assert annual_state_overhead(
        capacity_mw=1_000.0, ownership_share=1.0, rate_gbp_per_kw_year=3.0,
        component_type="StorageUnit", applies_to=("Generator", "StorageUnit")) == 3.0e6


# ── the account ──────────────────────────────────────────────────────────────


def _assets():
    common = {
        "model_year": 2030, "component_type": "Generator", "carrier": "solar_pv",
        "finance_technology": "Solar PV", "capacity_mw": 1_000.0, "dispatch_mwh": 1.0e6,
        "evidence_status": "resolved", "support_type": "merchant",
    }
    return pd.DataFrame([
        {**common, "asset_id": "new-solar", "finance_vintage": "new_build", "commissioning_year": 2027},
        {**common, "asset_id": "old-solar", "finance_vintage": "legacy", "commissioning_year": 2015},
    ])


def _totals(**overhead):
    snapshot = {
        "rate_basis": "real_CPI",
        "assumptions": {"public_coordination_state_equity_share": {"value": 0.3},
                        "public_borrowing_real": {"value": 0.0275}},
        "construction_risk_premium_by_rating": {"L-M": {"value": 0.0075}},
        "technologies": [{**ROW, "technology": "Solar PV", "risk_rating": "L-M"}],
    }
    scenarios = {"private": {}, "public_ownership": {"new_build_generation": "public",
                                                     "public_return": "public_rate"}}
    register, totals = calculate_financing_scenarios(
        _assets(), _costs(), snapshot, scenarios=scenarios,
        sensitivities={"structural": [], "one_at_a_time": [], "named": {}, "bounding_pair": False},
        price_year=2024, state_operating_overhead=dict(OVERHEAD, **overhead))
    return register, totals.set_index("case_id")


def test_the_overhead_is_a_line_of_required_revenue_and_only_on_the_public_case():
    register, totals = _totals()
    # 1 GW of state-owned new build at 3 GBP/kW/yr.  The legacy row stays
    # private in this bundle, so the state runs only what it owns.
    assert totals.loc["public_ownership", "annual_state_overhead_gbp"] == pytest.approx(3.0e6)
    assert totals.loc["private", "annual_state_overhead_gbp"] == 0.0
    by_case = register.set_index(["case_id", "asset_id"])["annual_state_overhead_gbp"]
    assert by_case[("public_ownership", "old-solar")] == 0.0
    # It raises the public case's required revenue, so it CUTS the financing
    # saving by its own size: this is the whole point of pricing it.
    without = _totals(enabled=False)[1]
    assert (totals.loc["public_ownership", "annual_required_revenue_gbp"]
            - without.loc["public_ownership", "annual_required_revenue_gbp"]) == pytest.approx(3.0e6)
    assert totals.loc["private", "annual_required_revenue_gbp"] == pytest.approx(
        without.loc["private", "annual_required_revenue_gbp"])
    assert (without.loc["public_ownership", "annual_financing_saving_vs_private_gbp"]
            - totals.loc["public_ownership", "annual_financing_saving_vs_private_gbp"]) == pytest.approx(3.0e6)


def test_the_acquisition_tariff_carries_the_overhead_as_its_own_line():
    """The tariff is summed from components, not from the required-revenue total.

    So the overhead needs a declared line of its own; without one, item OH1
    would apply to the market-bill propositions and vanish under P3, which is
    exactly where the state runs the most plant.
    """
    assert LINE_STATE_OVERHEAD in NEW_LINE_LABELS
    assert "OH1" in NEW_LINE_LABELS[LINE_STATE_OVERHEAD]


# ── F19 decision 6: unstaffed storage at its own rate ──────────────────────────


def test_unstaffed_storage_takes_its_own_rate_with_twins_in_proportion():
    from scripts.public_ownership.finance import STATE_OVERHEAD_STORAGE_DEFAULTS, state_overhead_carrier_rates

    rates = {value: state_overhead_carrier_rates(OVERHEAD, _s(state_overhead=value))
             for value in ("central", "low", "high")}
    assert rates["central"] == {"Battery": 1.5, "LAES": 1.5, "pumped_hydro": 1.5}
    assert (rates["low"]["Battery"], rates["high"]["Battery"]) == (0.5, 4.0)
    # In proportion to the fleet rate's twins: 1.0/3.0 and 8.0/3.0 of the central.
    assert rates["low"]["Battery"] / 1.5 == pytest.approx(1.0 / 3.0)
    assert rates["high"]["Battery"] / 1.5 == pytest.approx(8.0 / 3.0)
    # The 2050 (vintaged) register keeps the fleet rate; a disabled block has none.
    assert state_overhead_carrier_rates(OVERHEAD, _s(), vintaged=True) == {}
    assert state_overhead_carrier_rates({"enabled": False}, _s()) == {}
    # The code default equals the repository config (house rule).
    block = load_defaults()["public_ownership"]
    configured = block["state_operating_overhead"]["unstaffed_storage"]
    assert {k: (tuple(v) if isinstance(v, list) else v) for k, v in configured.items()} == dict(
        STATE_OVERHEAD_STORAGE_DEFAULTS)


def test_a_storage_carrier_row_carries_the_storage_rate_in_the_account():
    # The synthetic fleet is solar; naming its carrier as unstaffed storage moves its row to 1.5.
    register, totals = _totals(unstaffed_storage={"carriers": ["solar_pv"], "central_gbp_per_kw_year": 1.5,
                                                  "low_gbp_per_kw_year": 0.5, "high_gbp_per_kw_year": 4.0})
    assert totals.loc["public_ownership", "annual_state_overhead_gbp"] == pytest.approx(1.5e6)
    assert totals.loc["private", "annual_state_overhead_gbp"] == 0.0
