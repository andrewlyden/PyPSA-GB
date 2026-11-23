"""Item MR4: the Capacity Market contract book and auction model."""

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.market import capacity_market as cm

REGISTER = cm.REGISTER_CMU
ASSUMPTIONS = Path("data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumptions.json")
needs_register = pytest.mark.skipif(not REGISTER.exists(), reason="NESO Capacity Market Register not downloaded")


def test_demand_curve_passes_through_cap_net_cone_and_zero():
    curve = cm.DemandCurve(target_mw=40900, min_at_cap_mw=39400, max_at_zero_mw=42400, net_cone=49, price_cap=75)
    assert curve.price(39400) == pytest.approx(75) and curve.price(40900) == pytest.approx(49)
    assert curve.price(42400) == pytest.approx(0)
    for q in (39000, 39900, 40900, 41700, 43000):
        assert curve.quantity(curve.price(q)) == pytest.approx(min(max(q, 39400), 42400))


def test_auction_clears_where_supply_meets_the_curve():
    curve = cm.DemandCurve(40900, 39400, 42400, 49, 75)
    # 40 GW bids nothing, 5 GW bids uniformly up to the cap: S(p) = 40 + 5 p / 75.
    entered = pd.Series({"a": 40000.0, "b": 5000.0})
    result = cm.clear_auction(entered, {"a": (1.0, 1.0), "b": (0.0, 1.0)}, curve)
    p = result["price"]
    assert 40000 + 5000 * p / 75 == pytest.approx(curve.quantity(p), abs=1.0)
    # Abundant zero bids clear at zero; too little supply clears at the cap.
    assert cm.clear_auction(pd.Series({"a": 50000.0}), {"a": (1.0, 1.0)}, curve)["price"] == 0.0
    short = cm.clear_auction(pd.Series({"a": 30000.0}), {"a": (1.0, 1.0)}, curve)
    assert short["price"] == 75 and short["shortfall"]


def test_os_grid_reference_converts_to_metres():
    assert cm.os_grid_to_easting_northing("NO 396 371") == (339600, 737100)
    assert cm.os_grid_to_easting_northing("SE 989 059") == (498900, 405900)
    assert cm.os_grid_to_easting_northing("not a ref") is None


def test_base_year_and_cpi_indexation():
    assert cm.base_financial_year("2029-30 (T-4) Four Year Ahead Capacity Auction", "Winter 2024/2025") == 2024
    assert cm.base_financial_year("2026-27 (T-1) One Year Ahead Capacity Auction", None) == 2024
    cpi = pd.Series({2024: 133.9, 2025: 138.4})
    assert cm.cpi_financial_year(cpi, 2024) == pytest.approx(0.75 * 133.9 + 0.25 * 138.4)


@needs_register
def test_register_reproduces_the_published_auction_totals():
    register = cm.read_register()
    t4 = cm.entered_by_category(register, "2029-30 (T-4) Four Year Ahead Capacity Auction")
    assert t4["awarded_mw"].sum() == pytest.approx(40108.608, abs=5)      # Auction Monitor Report
    assert t4["entered_mw"].sum() == pytest.approx(44100, abs=200)       # 44.1 GW entered
    prior = cm.entered_by_category(register, "2028-29 (T-4) Four Year Ahead Capacity Auction")
    assert prior["awarded_mw"].sum() == pytest.approx(43055.073, abs=15)


@needs_register
def test_contract_book_counts_earlier_multi_year_agreements():
    agreements = cm.read_agreements()
    book = cm.contract_book(agreements, "2029/2030", cm.read_cpi())
    t4 = book["auction"].eq("2029-30 (T-4) Four Year Ahead Capacity Auction")
    assert book.loc[t4, "mw"].sum() == pytest.approx(40108.608, abs=5)
    assert book.loc[~t4, "mw"].sum() > 15000                               # 15-year and 3-year vintages
    # Older vintages are indexed up: a 2018-19 T-4 agreement is worth more than its 19.40 today.
    old = book[book["auction"].str.startswith("2018-19")]
    assert (old["payment_gbp"] / (old["mw"] * 1e3)).min() > 19.40 * 1.3


@needs_register
def test_auction_model_reproduces_both_held_t4_auctions():
    from scripts.market.build_capacity_market import auction_model

    model = auction_model(cm.read_register())
    assert model["checks"]["T-4 2029/30"]["modelled_price"] == pytest.approx(27.10, abs=1.5)
    assert model["checks"]["T-4 2028/29"]["modelled_price"] == pytest.approx(60.00, abs=1.0)
    # Without a fleet (the 2050 build) there is no 2030/31 projection.
    assert "t4_2030_31" not in model


def _fleet_paths():
    from scripts.market import build_capacity_market as bcm

    return bcm.FLEET_ASSETS, bcm.FLEET_INTERCONNECTORS, bcm.FLEET_ES1


needs_fleet = pytest.mark.skipif(not (REGISTER.exists() and all(p.exists() for p in _fleet_paths())),
                                 reason="CP30 2030 fleet inputs (cp30_2030_cp30_network asset inputs, cables, ES1) not built")


@needs_fleet
def test_the_bill_carries_the_fleet_auction_and_its_cases():
    """User decision (6 Oct 2026): the bill pins the 2030/31 T-4 cleared on the CP30 fleet, and each case."""
    from scripts.market import build_capacity_market as bcm

    register = cm.read_register()
    agreements, cpi = cm.read_agreements(register), cm.read_cpi()
    fleet = bcm.read_fleet_2030(*_fleet_paths())
    model = bcm.auction_model(register, fleet, agreements, cpi)
    cases = model["t4_2030_31"]
    assert set(cases) == set(bcm.CASES_2030)
    # More DSR, less price; lower storage factors or no new LDES, more.
    assert cases["dsr_high"]["price"] < cases["central"]["price"] < cases["dsr_low"]["price"]
    assert cases["storage_efc_low"]["price"] > cases["central"]["price"]
    assert cases["ldes_excluded"]["price"] > cases["central"]["price"]
    summary, _ = bcm.calendar_2030(agreements, cpi, model)
    sourced = json.loads(ASSUMPTIONS.read_text(encoding="utf-8"))["sourced_externals"]["capacity_market"]
    derivation = sourced["derivation"]
    assert sourced["price_year"] == 2025 and "rebasing" not in sourced
    assert derivation["agreement_cost_gbp"] == pytest.approx(summary["calendar_2030_cost_gbp"], rel=1e-6)
    assert derivation["recovery_denominator"] == "gsp_charging_volume_mwh" and "value" not in sourced
    assert derivation["t4_2030_31_central_price_gbp_per_kw_year"] == pytest.approx(cases["central"]["price"])
    assert derivation["t4_2030_31_central_awarded_mw"] == pytest.approx(cases["central"]["awarded_mw"])
    for label, cost in summary["calendar_2030_cost_by_t4_case_gbp"].items():
        assert derivation["calendar_2030_cost_by_t4_case_gbp"][label] == pytest.approx(cost, rel=1e-6)
        assert derivation["t4_2030_31_price_by_case_gbp_per_kw_year"][label] == pytest.approx(cases[label]["price"])
        if label != "central":
            assert derivation["sensitivity_cost_gbp"][f"t4_case_{label}"] == pytest.approx(cost, rel=1e-6)
    assert derivation["sensitivity_cost_gbp"]["t4_2030_31_at_2029_30_clearing_27_10"] == pytest.approx(
        summary["calendar_2030_cost_t4_at_2029_30_clearing_gbp"], rel=1e-6)
    block = summary["delivery_years"]["2030/2031"]
    for key in ("contract_book_gbp", "t4_modelled_gbp", "t1_gbp", "total_gbp"):
        assert derivation["delivery_year_2030_31_gbp"][key] == pytest.approx(block[key], rel=1e-6)


@needs_register
def test_household_peak_factor_is_ofgems_2024_ratio_and_the_bill_pins_it():
    from scripts.market.build_capacity_market import ofgem_domestic_peak_factor

    factor = ofgem_domestic_peak_factor(cm.read_agreements(), cm.read_cpi())
    # Households are concentrated in the 16:00-19:00 winter charging window.
    assert 1.2 < factor["factor"] < 1.5
    assert factor["ofgem_allowance_gbp_per_3100kwh_nominal"] == pytest.approx(20.78, abs=0.05)
    sourced = json.loads(ASSUMPTIONS.read_text(encoding="utf-8"))["sourced_externals"]["capacity_market"]
    assert sourced["derivation"]["household_allocation_factor"] == pytest.approx(factor["factor"], rel=1e-5)


def test_cm_revenue_goes_to_merchant_units_capped_at_the_contract_book():
    rates = pd.DataFrame({"carrier": ["CCGT", "long_duration_storage", "solar_pv"],
                          "gbp_per_kw_connection_year": [40.0, 45.0, 3.5],
                          "connection_mw_2029_30": [30000.0, 2000.0, 600.0]})
    assets = pd.DataFrame({
        "asset_id": ["g1", "g2", "ldes", "pv_m", "pv_cfd", "eu"],
        "carrier": ["CCGT", "CCGT", "LAES", "solar_pv", "solar_pv", "EU_market"],
        "support_type": ["merchant", "merchant", None, "merchant", "CfD", None],
        "capacity_mw": [10000.0, 10000.0, 8000.0, 6000.0, 20000.0, 5000.0]})
    out = cm.attach_capacity_market_revenue(assets, rates).set_index("asset_id")
    # Gas: model 20 GW under 30 GW contracted, so the full rate.
    assert out.loc["g1", "capacity_market_revenue_gbp"] == pytest.approx(10000 * 1e3 * 40.0)
    # LDES: 8 GW in the model, 2 GW contracted: a quarter participates.
    assert out.loc["ldes", "capacity_market_revenue_gbp"] == pytest.approx(8000 * 1e3 * 45.0 * 0.25)
    # Merchant solar pro rata to the 0.6 GW contracted; CfD solar and foreign plant earn nothing.
    assert out.loc["pv_m", "capacity_market_revenue_gbp"] == pytest.approx(600 * 1e3 * 3.5)
    assert out.loc["pv_cfd", "capacity_market_revenue_gbp"] == 0.0 and out.loc["eu", "capacity_market_revenue_gbp"] == 0.0


def test_legacy_valuation_counts_capacity_market_revenue_as_operating_cash():
    from scripts.public_ownership.legacy_valuation import RO_SHARE_ALL, legacy_cash_flow_profile, settings_from_config

    rows = pd.DataFrame({
        "model_year": [2030], "asset_id": ["g1"], "carrier": ["CCGT"], "component_type": ["Generator"],
        "support_type": ["merchant"], "cfd_round": [None], "cfd_award_year": [None], "capacity_mw": [100.0],
        "commissioning_year": [2000], "market_revenue_gbp": [5e6], "cfd_support_gbp": [0.0],
        "cfd_curtailment_compensation_gbp": [0.0], "roc_support_gbp": [0.0], "fuel_carbon_gbp": [4e6],
        "annual_fixed_om_gbp": [2e6], "wacc": [0.08], "capacity_market_revenue_gbp": [3e6]})
    settings = {**settings_from_config(None), "horizon_year": 2032}
    flows, _ = legacy_cash_flow_profile(rows, settings, pd.DataFrame({RO_SHARE_ALL: 0.0}, index=range(2030, 2033)))
    assert flows["net_operating_cash"][0, 0] == pytest.approx(5e6 + 3e6 - 4e6 - 2e6)
