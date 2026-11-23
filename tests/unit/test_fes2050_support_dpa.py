"""Task Y11 (FES 2050 plan): DPAs for gas CCS and hydrogen power, BECCS, LDES cap and floor."""

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from scripts.market import cfd_strikes as cfd50
from scripts.public_ownership import support_regime_2050 as sr
from scripts.public_ownership import vintage_costs as vc
from scripts.public_ownership.finance import capital_recovery_factor

needs_snapshot = pytest.mark.skipif(not Path(cfd50.DEFAULT_SNAPSHOT).exists(), reason="workbook snapshot not extracted")


@pytest.fixture(scope="module")
def snapshot():
    return cfd50.load_snapshot()


@pytest.fixture(scope="module")
def params():
    return sr.load_parameters()


@needs_snapshot
def test_ccs_availability_rate_is_cost_plus_the_private_rate(snapshot, params):
    c = sr.dpa_contract("CCS Gas", 2040, 900.0, snapshot=snapshot, params=params)
    cost = vc.costs_for("Gas generation - CCUS - mature", 2040)
    rate = cfd50.private_rate("Gas generation - CCUS - mature", 2040, snapshot=snapshot)["rate"]
    # Insurance on the register's basis (Y14-prep): DESNZ insurance out, D4(a) 0.4% of capex in.
    fixed = cost["capex_power_gbp_per_kw"] * capital_recovery_factor(rate, cost["economic_life_years"]) \
        + cost["fixed_om_gbp_per_kw_year"] - cost["insurance_gbp_per_kw_year"] \
        + 0.004 * cost["capex_power_gbp_per_kw"] + 4.4
    assert c.fixed_cost_gbp_per_kw_year == pytest.approx(fixed)
    assert c.apr_gbp_per_kw_year == pytest.approx(fixed / 0.90)
    assert c.term_years == 15 and c.live_share == 1.0
    # The DESNZ EGC 2025 efficiency (user, 26 Oct 2026).
    assert c.efficiency == pytest.approx(0.47) and c.variable_om_gbp_per_mwh == pytest.approx(18.09)
    assert c.residual_emission_tco2_per_mwh == pytest.approx(0.183 * 0.1 / 0.47)
    # 15-year terms: a 2035 build is half live in 2050, a 2030 build (and NZT Power, 2028) has ended.
    assert sr.dpa_contract("CCS Gas", 2035, 1.0, snapshot=snapshot, params=params).live_share == pytest.approx(0.5)
    assert sr.dpa_contract("CCS Gas", 2030, 1.0, snapshot=snapshot, params=params).live_share == 0.0


@needs_snapshot
def test_hydrogen_dpa_blends_ccht_and_ocht(snapshot, params):
    h = sr.dpa_contract("Hydrogen", 2045, 1.0, snapshot=snapshot, params=params)
    assert sum(sr.HYDROGEN_TECHNOLOGY_SPLIT.values()) == pytest.approx(1.0)
    parts = []
    for technology, share in sr.HYDROGEN_TECHNOLOGY_SPLIT.items():
        cost = vc.costs_for(technology, 2045)
        rate = cfd50.private_rate(technology, 2045, snapshot=snapshot)["rate"]
        parts.append(share * (cost["capex_power_gbp_per_kw"] * capital_recovery_factor(rate, cost["economic_life_years"])
                              + cost["fixed_om_gbp_per_kw_year"] - cost["insurance_gbp_per_kw_year"]
                              + 0.004 * cost["capex_power_gbp_per_kw"] + 4.4))
    assert h.fixed_cost_gbp_per_kw_year == pytest.approx(sum(parts))


def _contract(carrier="CCS Gas", **changes):
    base = sr.DPAContract(carrier=carrier, build_year=2040, capacity_mw=100.0, apr_gbp_per_kw_year=250.0,
                          expected_availability=0.9, term_years=15, live_share=1.0, efficiency=0.5,
                          variable_om_gbp_per_mwh=15.0, residual_emission_tco2_per_mwh=0.0366 if carrier == "CCS Gas" else 0.0,
                          margin_gbp_per_mwh=1.0, gainshare_share=0.5, gainshare_threshold_gbp_per_kw_year=0.0,
                          private_rate=0.08, fixed_cost_gbp_per_kw_year=225.0, source="synthetic")
    return replace(base, **changes)


def test_dpa_settlement_on_synthetic_hours(params):
    index = pd.date_range("2050-01-01", periods=3, freq="h")
    price = pd.Series([150.0, 60.0, 100.0], index=index)
    dispatch = pd.Series([100.0, 0.0, 50.0], index=index)
    c = _contract()
    # High carbon: CCS is cheaper than the unabated reference, so no variable payment.
    out = sr.dpa_settlement(c, dispatch, price, gas=23.64, carbon=110.8, params=params)
    srmc = 23.64 / 0.5 + 110.8 * 0.0366 + 15.0
    assert out["variable_payment_gbp"] == 0.0
    assert out["availability_payment_gbp"] == pytest.approx(250.0 * 100e3 * 0.9)
    margin = (150 - srmc) * 100 + (100 - srmc) * 50
    assert out["energy_margin_gbp"] == pytest.approx(margin)
    assert out["gainshare_gbp"] == pytest.approx(0.5 * margin)
    assert out["net_support_cost_gbp"] == pytest.approx(out["availability_payment_gbp"] - 0.5 * margin)
    # Zero carbon: the reference CCGT is cheaper, and the top-up keeps CCS just ahead of it.
    zero = sr.dpa_settlement(c, dispatch, price, gas=23.64, carbon=0.0, params=params)
    reference = 23.64 / 0.52 + 2.0
    assert zero["variable_payment_gbp"] == pytest.approx((23.64 / 0.5 + 15.0 - reference + 1.0) * 150)
    # A half-live contract pays half; measured availability replaces the expected one.
    half = sr.dpa_settlement(replace(c, live_share=0.5), dispatch, price, gas=23.64, carbon=110.8,
                             availability=0.8, params=params)
    assert half["availability_payment_gbp"] == pytest.approx(250.0 * 100e3 * 0.8 * 0.5)


def test_hydrogen_dpa_needs_the_hydrogen_price(params):
    index = pd.date_range("2050-01-01", periods=2, freq="h")
    c = _contract("Hydrogen", efficiency=0.55, variable_om_gbp_per_mwh=6.0)
    with pytest.raises(ValueError):
        sr.dpa_settlement(c, pd.Series([10.0, 10.0], index=index), pd.Series([200.0, 200.0], index=index),
                          gas=23.64, carbon=110.8, params=params)
    h2 = pd.Series([80.0, 40.0], index=index)
    # Hydrogen power has no variable payment (user, 26 Oct 2026): availability and gainshare only.
    assert not sr.has_variable_payment("Hydrogen", params) and sr.has_variable_payment("CCS Gas", params)
    out = sr.dpa_settlement(c, pd.Series([10.0, 10.0], index=index), pd.Series([200.0, 200.0], index=index),
                            gas=23.64, carbon=110.8, hydrogen_price=h2, params=params)
    assert out["variable_payment_gbp"] == 0.0
    assert out["short_run_cost_gbp"] == pytest.approx(10 * (80 / 0.55 + 6) + 10 * (40 / 0.55 + 6))
    # A parameter file without the key gives every DPA carrier the payment (the earlier rule).
    legacy = {**params, "dpa": {k: v for k, v in params["dpa"].items() if k != "variable_payment_carriers"}}
    out = sr.dpa_settlement(c, pd.Series([10.0, 10.0], index=index), pd.Series([200.0, 200.0], index=index),
                            gas=23.64, carbon=110.8, hydrogen_price=h2, params=legacy)
    reference = (23.64 + 110.8 * 0.183) / 0.52 + 2.0
    top_up = [max(80 / 0.55 + 6 - reference + 1, 0.0), max(40 / 0.55 + 6 - reference + 1, 0.0)]
    assert out["variable_payment_gbp"] == pytest.approx(10 * sum(top_up))


@needs_snapshot
def test_beccs_strike_nets_the_removals_leg(snapshot, params):
    row = sr.beccs_strike(2040, snapshot=snapshot, params=params)
    captured = 0.403 * 0.9 / 0.35
    assert row["captured_tco2_per_mwh"] == pytest.approx(captured)
    assert row["other_revenue_gbp_per_mwh"] == pytest.approx(-captured * 110.8)
    assert row["fuel_gbp_per_mwh"] == pytest.approx(45.0 / 0.35)
    gross = cfd50.levelised_strike("Biomass - with CCUS - mature", 2040, snapshot=snapshot,
                                   fuel_gbp_per_mwh=45.0 / 0.35)
    assert row["strike_gbp_per_mwh_2024"] == pytest.approx(gross["strike_gbp_per_mwh_2024"] - captured * 110.8)
    assert row["term_years"] == 15


@needs_snapshot
def test_ldes_cap_and_floor(snapshot, params):
    c = sr.ldes_contract("LAES", 2040, 100.0, snapshot=snapshot, params=params)
    assert c.floor_rate == 0.045 and c.cap_rate > c.floor_rate and c.cap_gbp > c.floor_gbp
    assert c.term_years == 25 and c.live_share == 1.0
    assert sr.ldes_contract("Pumped Storage Hydroelectricity", 2010, 100.0, snapshot=snapshot,
                            params=params).live_share == 0.0          # legacy pumped hydro is merchant
    low = sr.ldes_settlement(c, market_revenue_gbp=0.5 * c.floor_gbp, capacity_market_revenue_gbp=0.1 * c.floor_gbp,
                             params=params)
    assert low["floor_topup_gbp"] == pytest.approx(0.4 * c.floor_gbp) and low["cap_return_gbp"] == 0.0
    high = sr.ldes_settlement(c, market_revenue_gbp=c.cap_gbp + 1e6, params=params)
    assert high["cap_return_gbp"] == pytest.approx(1e6) and high["net_support_cost_gbp"] == pytest.approx(-1e6)


def test_bill_line_map_names_every_scheme():
    schemes = " ".join(row["scheme"] for row in sr.bill_line_map())
    for name in ("CfD", "DPA", "BECCS", "Capacity Market", "LDES", "Sizewell C", "Renewables Obligation"):
        assert name in schemes
    for row in sr.bill_line_map():
        assert row["bill_line"] and row["denominator"]


def test_parameters_carry_sources(params):
    for block in ("dpa", "beccs", "ldes_cap_and_floor"):
        for key, value in params[block].items():
            if isinstance(value, dict) and "value" in value:
                assert value.get("source") and value.get("status"), f"{block}.{key}"
