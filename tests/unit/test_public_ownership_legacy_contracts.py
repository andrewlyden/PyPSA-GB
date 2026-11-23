"""Item G3, lever L19: what happens to the support contracts of acquired legacy plant.

Decision 8 (22 September 2026): build all three treatments. The user placed L19
inside the acquisition tariff, where every legacy-acquiring case lives, and valued
the residual at owners' WACC:

* ``honoured`` (today): the purchase price includes the CfD top-up and ROC income
  streams, discounted at the public rate; the contracts become intra-public.
* ``cancelled_residual_on_bills``: the price excludes those streams, and the
  former owners are paid a residual, the streams' value at their own WACC,
  annuitised on the same public-rate term, as its own tariff line.
* ``cancelled_residual_on_exchequer``: the same, with the residual off the bill
  and reported as a transfer from the Exchequer.

The identity the item exists to hold: cancelling saves the rate differential on
the streams and nothing else, so the saving is not counted twice. Under the
``market_value`` basis the price already discounts at owners' WACC, and
cancelling on bills moves the tariff by exactly zero.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.public_ownership.calculate_financing_scenarios import calculate_financing_scenarios
from scripts.public_ownership.finance import capital_recovery_factor
from scripts.public_ownership.levers import (
    DEFAULT_SETTINGS, LEVERS, LeverConfigError, parse_scenarios, parse_sensitivities,
)
from scripts.public_ownership.nationalised_tariff import (
    LINE_CANCELLED_RESIDUAL, LINE_LEGACY_RENEWABLES, revenue_requirement,
)

from tests.unit.test_public_ownership_cfd_repricing import _costs
from tests.unit.test_public_ownership_legacy_valuation import _stations
from tests.unit.test_public_ownership_nationalised_case import (
    PUBLIC_RATE, _acquisition, _register_inputs, _snapshot_with_public_rate,
)

L19 = "legacy_contracts_under_acquisition"
NO_OVERHEAD = {"enabled": False}


# ── the lever ────────────────────────────────────────────────────────────────

def test_the_lever_has_three_treatments_and_honours_by_default():
    lever = LEVERS[L19]
    assert lever.values == ("honoured", "cancelled_residual_on_bills", "cancelled_residual_on_exchequer")
    assert lever.default == "honoured" and lever.register
    assert {lever.label(v) for v in lever.values} == {
        "contracts_honoured", "contracts_cancelled_on_bills", "contracts_cancelled_on_exchequer"}


def _scenarios():
    return parse_scenarios({
        "private": {},
        "public_ownership": {"new_build_generation": "public"},
    })


def test_a_cancelled_treatment_needs_an_acquisition():
    with pytest.raises(LeverConfigError, match="acquire"):
        parse_sensitivities({"named": {"cancel_nothing": {
            "scenario": "public_ownership", L19: "cancelled_residual_on_bills"}}}, _scenarios())


def test_full_nationalisation_is_exempt():
    """The state has bought the counterparty; there is no contract left to cancel."""
    with pytest.raises(LeverConfigError, match="full nationalisation"):
        parse_sensitivities({"named": {"cancel_under_p3": {
            "scenario": "public_ownership", "legacy_fleet": "acquire_all",
            "pre_2025_cfd_capacity": "public_repriced", L19: "cancelled_residual_on_bills"}}}, _scenarios())
    ok = parse_sensitivities({"named": {"cancel_legacy": {
        "scenario": "public_ownership", "legacy_fleet": "acquire_valued_group",
        L19: "cancelled_residual_on_bills"}}}, _scenarios())
    assert ok["named"]["cancel_legacy"]["settings"][L19] == "cancelled_residual_on_bills"


# ── the accounting ───────────────────────────────────────────────────────────

def _specs(basis, *treatments):
    from scripts.public_ownership.levers import CaseSpec
    specs = [CaseSpec("private", "private", dict(DEFAULT_SETTINGS))]
    for treatment in treatments:
        settings = dict(DEFAULT_SETTINGS, new_build_generation="public", legacy_fleet="acquire_valued_group",
                        purchase_price_basis=basis, **{L19: treatment})
        specs.append(CaseSpec(f"public_ownership@{treatment}", "public_ownership", settings, treatment))
    return specs


def _register(basis="state_present_value"):
    register, totals = calculate_financing_scenarios(
        _register_inputs(), _costs(), _snapshot_with_public_rate(),
        case_specs=_specs(basis, "honoured", "cancelled_residual_on_bills", "cancelled_residual_on_exchequer"),
        acquisition=_acquisition(), ro_stations=_stations(), state_operating_overhead=NO_OVERHEAD)
    return register, totals


def _valued(register, case):
    rows = register.loc[register["case_id"].eq(f"public_ownership@{case}")].set_index("asset_id")
    return rows.loc[rows["legacy_scope_group"].astype(str).eq("valued")]


def test_cancelling_removes_the_contract_streams_from_the_price():
    register, _ = _register()
    honoured, cancelled = _valued(register, "honoured"), _valued(register, "cancelled_residual_on_bills")
    streams_state = cancelled["pv_state_cancelled_streams_gbp"]
    assert (streams_state > 0).any()
    expected = (honoured["acquisition_price_gbp"] - streams_state).clip(lower=0.0)
    assert np.allclose(cancelled["acquisition_price_gbp"], expected)
    assert (_valued(register, "honoured")["annual_cancelled_contract_residual_gbp"].fillna(0.0) == 0.0).all()


def test_the_residual_is_the_streams_at_owners_wacc_annuitised_at_the_public_rate():
    register, _ = _register()
    cancelled = _valued(register, "cancelled_residual_on_bills")
    # F28 (e): each row repaid over its own remaining life.
    crf = cancelled["acquisition_term_years_applied"].astype(float).map(lambda n: capital_recovery_factor(PUBLIC_RATE, n))
    assert np.allclose(cancelled["annual_cancelled_contract_residual_gbp"],
                       cancelled["pv_market_cancelled_streams_gbp"] * crf)
    # Owners discount at a higher rate than the state, so the residual is smaller.
    assert (cancelled["pv_market_cancelled_streams_gbp"] <= cancelled["pv_state_cancelled_streams_gbp"] + 1e-6).all()
    assert set(cancelled["cancelled_residual_route"]) == {"bills"}
    assert set(_valued(register, "cancelled_residual_on_exchequer")["cancelled_residual_route"]) == {"exchequer"}


def _tariff(register, case):
    lines, diagnostics = revenue_requirement(register, case_id=f"public_ownership@{case}")
    return lines, diagnostics


def test_the_tariff_saving_is_the_rate_differential_only():
    register, _ = _register()
    honoured, _ = _tariff(register, "honoured")
    on_bills, _ = _tariff(register, "cancelled_residual_on_bills")
    cancelled = _valued(register, "cancelled_residual_on_bills")
    # F28 (e): each row repaid over its own remaining life.
    crf = cancelled["acquisition_term_years_applied"].astype(float).map(lambda n: capital_recovery_factor(PUBLIC_RATE, n))
    differential = float(((cancelled["pv_state_cancelled_streams_gbp"]
                           - cancelled["pv_market_cancelled_streams_gbp"]) * crf).sum())
    assert differential > 0
    assert sum(honoured.values()) - sum(on_bills.values()) == pytest.approx(differential, rel=1e-9)
    assert on_bills[LINE_CANCELLED_RESIDUAL] > 0 and honoured[LINE_CANCELLED_RESIDUAL] == 0.0


def test_the_exchequer_variant_moves_the_residual_off_the_bill_and_reports_it():
    register, _ = _register()
    on_bills, _ = _tariff(register, "cancelled_residual_on_bills")
    on_exchequer, diagnostics = _tariff(register, "cancelled_residual_on_exchequer")
    assert on_exchequer[LINE_CANCELLED_RESIDUAL] == 0.0
    assert diagnostics["cancelled_contract_residual_exchequer_transfer_gbp"] == pytest.approx(
        on_bills[LINE_CANCELLED_RESIDUAL])
    assert on_exchequer[LINE_LEGACY_RENEWABLES] == pytest.approx(on_bills[LINE_LEGACY_RENEWABLES])


def test_at_market_value_cancelling_on_bills_changes_nothing():
    """The price already discounts at owners' WACC, so there is no differential."""
    register, _ = _register("market_value")
    honoured, _ = _tariff(register, "honoured")
    on_bills, _ = _tariff(register, "cancelled_residual_on_bills")
    assert sum(on_bills.values()) == pytest.approx(sum(honoured.values()), rel=1e-9)


def test_a_cancelled_treatment_on_an_apportioned_price_is_refused():
    """A book value (or any price not built from the streams) cannot say which part of it was the contracts."""
    with pytest.raises(ValueError, match="cancelled"):
        calculate_financing_scenarios(
            _register_inputs(), _costs(), _snapshot_with_public_rate(),
            case_specs=_specs("book_value", "cancelled_residual_on_bills"),
            acquisition=_acquisition(), ro_stations=_stations(), state_operating_overhead=NO_OVERHEAD)
