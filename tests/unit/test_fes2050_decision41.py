"""FES 2050 decision 41: a self-financing private counterfactual.

(a) expired contracts' capital is sunk; the contract-term-with-tail strike basis and the
register's capital on it; (b) the break-even calibration; (c) Sizewell C's RAB allowance,
and the bill and tariff lines that read it.
"""

import json

import numpy as np
import pandas as pd
import pytest

from scripts.public_ownership import vintage_capital_2050 as vk
from scripts.public_ownership.calculate_financing_scenarios import (
    book_rab_allowed_revenue, calculate_financing_scenarios, rab_allowed_revenue_support, vintage_capital_options,
    vintage_cost_registry,
)
from scripts.public_ownership.finance import capital_recovery_factor
from scripts.public_ownership.vintage_costs import costs_for
from tests.helpers import requires_outputs

PRIVATE = {"Offshore wind": 0.0827, "Onshore wind": 0.0747, "Large-scale nuclear": 0.07}
SCENARIOS = {
    "private": {},
    "public_ownership": {"new_build_generation": "public", "public_return": "public_rate",
                         "public_rate_basis": "real_cpi"},
}
INSURANCE = {"insurance": {"enabled": True, "share_of_capex_per_year": 0.004,
                           "gbp_per_kw_year_by_technology": {"Offshore wind": 11.22},
                           "excluded_technologies": ["Large-scale nuclear", "Lithium batteries"]}}
NO_OVERHEAD = {"enabled": False}


def _snapshot():
    return {"rate_basis": "real_CPI",
            "assumptions": {"public_coordination_state_equity_share": {"value": 0.30},
                            "public_borrowing_real": {"value": 0.02745}},
            "technologies": [{"technology": t, "risk_rating": "Medium",
                              "wacc": {"private": r, "public_coordination": r - 0.008,
                                       "public_ownership_new_build": 0.0325, "public_ownership_legacy": 0.02745}}
                             for t, r in PRIVATE.items()]}


def _capture(tmp_path, onshore=37.58):
    frame = pd.DataFrame({"carrier": ["wind_offshore", "wind_onshore", "nuclear"],
                          "capture_gbp_per_mwh": [24.45, onshore, 35.38], "fuel_carbon_gbp_per_mwh": [0.0, 0.0, 10.0],
                          "load_factor": [0.42, 0.19, 0.75], "dispatch_mwh": [1.0, 1.0, 1.0],
                          "capacity_mw": [1.0, 1.0, 1.0], "market_revenue_gbp": [1.0, 1.0, 1.0], "source": "test"})
    path = tmp_path / "capture.csv"
    frame.to_csv(path, index=False)
    return path


def _assets():
    rows = [
        # A live post-AR7 onshore CfD, a part-live one, an expired-contract vintage, a legacy row,
        # and Sizewell C on its RAB.
        dict(asset_id="ON_2040", carrier="wind_onshore", finance_technology="Onshore wind",
             stock_flow_technology="Onshore wind", build_year=2040, support_type="CfD", cfd_round="PAR7_2040",
             support_basis="post_ar7_modelled", cfd_live_share=1.0, cfd_contract_id=None),
        dict(asset_id="ON_2031", carrier="wind_onshore", finance_technology="Onshore wind",
             stock_flow_technology="Onshore wind", build_year=2031, support_type="CfD", cfd_round="PAR7_2031",
             support_basis="post_ar7_modelled", cfd_live_share=0.5, cfd_contract_id=None),
        dict(asset_id="ON_2028", carrier="wind_onshore", finance_technology="Onshore wind",
             stock_flow_technology="Onshore wind", build_year=2028, support_type="merchant", cfd_round=None,
             support_basis="contract_expired_merchant", cfd_live_share=np.nan, cfd_contract_id=None),
        dict(asset_id="ON_2020", carrier="wind_onshore", finance_technology="Onshore wind",
             stock_flow_technology="Onshore wind", build_year=2020, support_type="merchant", cfd_round=None,
             support_basis="contract_expired_merchant", cfd_live_share=np.nan, cfd_contract_id=None),
        dict(asset_id="SZC_2039", carrier="nuclear", finance_technology="Large-scale nuclear",
             stock_flow_technology="Large-scale nuclear", build_year=2039, support_type="RAB", cfd_round="SZC_RAB",
             support_basis="known_contract", cfd_live_share=1.0, cfd_contract_id="SIZEWELL_C#1"),
    ]
    frame = pd.DataFrame(rows)
    frame["capacity_mw"] = 100.0
    frame["model_year"] = 2050
    frame["component_type"] = "Generator"
    frame["finance_vintage"] = np.where(frame["build_year"] < 2025, "legacy", "new_build")
    frame["commissioning_year"] = frame["build_year"].astype(float)
    frame["dispatch_mwh"] = frame["capacity_mw"] * 3000.0
    frame["evidence_status"] = "resolved"
    frame["contract_vintage"] = np.where(frame["support_type"].eq("CfD"), "post_policy", "not_applicable")
    frame["market_revenue_gbp"] = 5.0e6
    frame["support_revenue_gbp"] = 0.0
    frame["fuel_carbon_gbp"] = 0.0
    return frame


def _specs():
    from scripts.public_ownership.levers import expand_cases, parse_scenarios, parse_sensitivities

    bundles = parse_scenarios(SCENARIOS)
    return expand_cases(bundles, parse_sensitivities({"bounding_pair": False}, bundles))


def _run(assets, vintage_capital=None, snapshot=None):
    costs = pd.read_csv("data/public_ownership/financing/technology_costs.csv")
    register, _ = calculate_financing_scenarios(
        assets, costs, snapshot or _snapshot(), price_year=2024, state_operating_overhead=NO_OVERHEAD, generator_costs=INSURANCE,
        case_specs=_specs(), vintage_costs=vintage_cost_registry(), vintage_capital=vintage_capital)
    return register


def _basis(tmp_path, mode=vk.CONTRACT_TERM_WITH_TAIL):
    block = {vk.CONFIG_KEY: {"amortisation": mode, "capture_prices": str(_capture(tmp_path))},
             "generator_costs": INSURANCE}
    return vintage_capital_options(block)


# ── the basis ────────────────────────────────────────────────────────────────

def test_without_the_block_the_basis_is_economic_life_and_off():
    basis = vk.basis_from_config({})
    assert basis["amortisation"] == vk.ECONOMIC_LIFE and not basis["enabled"]
    assert vintage_capital_options({}) is None
    with pytest.raises(vk.VintageCapitalError):
        vk.basis_from_config({vk.CONFIG_KEY: {"amortisation": "contract_term_with_tail"}})
    with pytest.raises(vk.VintageCapitalError):
        vk.basis_from_config({vk.CONFIG_KEY: {"amortisation": "life"}})


def test_the_tail_value_is_the_margin_over_the_years_after_the_term():
    inputs = {"margin_gbp_per_kw_year": 10.0, "life_years": 30.0, "term_years": 20.0, "load_factor": 0.4}
    r = 0.07
    pv = sum(10.0 / (1 + r) ** t for t in range(21, 31))
    assert vk.tail_value_per_kw(inputs, r) == pytest.approx(pv)
    assert vk.tail_credit_gbp_per_mwh(inputs, r) == pytest.approx(pv * capital_recovery_factor(r, 20) / (8.76 * 0.4))
    assert vk.tail_value_per_kw(dict(inputs, margin_gbp_per_kw_year=0.0), r) == 0.0
    assert vk.tail_value_per_kw(dict(inputs, life_years=20.0), r) == 0.0


@requires_outputs("resources/public_ownership/workbook_v5_assumptions.yaml")
def test_the_central_strike_is_the_contract_term_strike_less_the_tail_credit(tmp_path):
    from scripts.market import cfd_strikes_2050 as cfd50

    snapshot, capture = cfd50.load_snapshot(), vk.load_capture_prices(_capture(tmp_path))
    term = vk.strike("Onshore wind", 2040, amortisation=vk.CONTRACT_TERM, snapshot=snapshot, capture=capture)
    tail = vk.strike("Onshore wind", 2040, amortisation=vk.CONTRACT_TERM_WITH_TAIL, snapshot=snapshot, capture=capture)
    life = vk.strike("Onshore wind", 2040, amortisation=vk.ECONOMIC_LIFE, snapshot=snapshot, capture=capture)
    assert tail["tail_credit_gbp_per_mwh"] > 0
    assert tail["strike_gbp_per_mwh_2024"] == pytest.approx(term["strike_gbp_per_mwh_2024"] - tail["tail_credit_gbp_per_mwh"])
    assert life["strike_gbp_per_mwh_2024"] == pytest.approx(
        cfd50.levelised_strike("Onshore wind", 2040, snapshot=snapshot)["strike_gbp_per_mwh_2024"])
    # Offshore's tail earns nothing at a 24.45 capture price: the strike is the contract-term one.
    off = vk.strike("Offshore wind", 2040, amortisation=vk.CONTRACT_TERM_WITH_TAIL, snapshot=snapshot, capture=capture)
    assert off["tail_credit_gbp_per_mwh"] == 0.0


# ── (a) the register on the term basis ───────────────────────────────────────

def test_the_economic_life_basis_leaves_the_register_unchanged(tmp_path):
    plain = _run(_assets())
    same = _run(_assets(), vintage_capital=_basis(tmp_path, vk.ECONOMIC_LIFE))
    pd.testing.assert_frame_equal(plain, same)


def test_expired_contracts_carry_no_capital_and_live_ones_the_term_annuity(tmp_path):
    basis = _basis(tmp_path)
    register = _run(_assets(), vintage_capital=basis)
    plain = _run(_assets())
    for case in ("private", "public_ownership"):
        rows = register.loc[register["case_id"].eq(case)].set_index("asset_id")
        before = plain.loc[plain["case_id"].eq(case)].set_index("asset_id")
        # (a): the 2028 vintage's contract has ended: capital and its tax are zero, O&M is kept.
        assert rows.loc["ON_2028", "annual_capital_recovery_gbp"] == 0.0
        assert rows.loc["ON_2028", "annual_required_revenue_gbp"] == pytest.approx(
            before.loc["ON_2028", "annual_required_revenue_gbp"] - before.loc["ON_2028", "annual_capital_recovery_gbp"])
        # The legacy row and Sizewell C's RAB keep the register's existing charge.
        for asset in ("ON_2020", "SZC_2039"):
            assert rows.loc[asset, "annual_capital_recovery_gbp"] == pytest.approx(
                before.loc[asset, "annual_capital_recovery_gbp"])
        # A live contract: (capex - tail value) x CRF(r, 20) for its live share.
        capture = vk.load_capture_prices(basis["capture_prices"])
        for asset, year, live in (("ON_2040", 2040, 1.0), ("ON_2031", 2031, 0.5)):
            r = float(rows.loc[asset, "wacc"])
            inputs = vk.tail_inputs("Onshore wind", year, capture=capture, insurance_config=INSURANCE["insurance"])
            capex = 100.0 * 1000.0 * costs_for("Onshore wind", year)["capex_power_gbp_per_kw"]
            expected = live * (capex - vk.tail_value_per_kw(inputs, r) * 100.0 * 1000.0) * capital_recovery_factor(r, 20)
            assert rows.loc[asset, "annual_capital_recovery_gbp"] == pytest.approx(expected), (case, asset)


@requires_outputs("resources/public_ownership/workbook_v5_assumptions.yaml")
def test_an_expired_row_embeds_no_corporation_tax_and_a_live_row_the_term_basis_tax(tmp_path):
    from scripts.market.cfd_strikes_2050 import load_snapshot

    snapshot = load_snapshot()
    register = _run(_assets(), vintage_capital=_basis(tmp_path), snapshot=snapshot)
    plain = _run(_assets(), snapshot=snapshot)
    owned = register.loc[register["case_id"].eq("public_ownership")].set_index("asset_id")
    before = plain.loc[plain["case_id"].eq("public_ownership")].set_index("asset_id")
    assert before.loc["ON_2028", "corporation_tax_forgone_gbp"] > 0.0
    assert owned.loc["ON_2028", "corporation_tax_forgone_gbp"] == 0.0
    assert 0.0 < owned.loc["ON_2040", "corporation_tax_forgone_gbp"]
    assert owned.loc["SZC_2039", "corporation_tax_forgone_gbp"] == pytest.approx(
        before.loc["SZC_2039", "corporation_tax_forgone_gbp"], nan_ok=True)


# ── (c) Sizewell C's RAB ─────────────────────────────────────────────────────

def test_sizewell_c_is_paid_its_allowed_revenue_in_every_case():
    register = _run(_assets())
    private = register.loc[register["case_id"].eq("private")].set_index("asset_id")
    owned = register.loc[register["case_id"].eq("public_ownership")].set_index("asset_id")
    assert private.loc["SZC_2039", "producer_rent_gbp"] == pytest.approx(0.0, abs=1e-3)
    allowed = private.loc["SZC_2039", "annual_required_revenue_gbp"] - private.loc["SZC_2039", "market_revenue_gbp"]
    assert private.loc["SZC_2039", "rab_support_gbp"] == pytest.approx(allowed)
    assert owned.loc["SZC_2039", "support_revenue_gbp"] == pytest.approx(allowed)
    assert owned.loc["SZC_2039", "rab_support_gbp"] == pytest.approx(allowed)
    assert (register.loc[register["support_type"].ne("RAB"), "rab_support_gbp"] == 0.0).all()


def test_a_register_without_rab_rows_is_returned_unchanged():
    frame = pd.DataFrame({"case_id": ["private"], "asset_id": ["A"], "support_type": ["CfD"],
                          "producer_rent_gbp": [-1.0], "support_revenue_gbp": [0.0], "ownership_share": [0.0],
                          "public_surplus_before_recycling_gbp": [0.0]})
    support = rab_allowed_revenue_support(frame)
    assert support.empty
    pd.testing.assert_frame_equal(book_rab_allowed_revenue(frame, support), frame)


def test_the_bill_line_reads_the_register_allowance_from_the_public_account(tmp_path):
    from scripts.consumer_bill.project_future_bill import register_booked_cost

    path = tmp_path / "account.json"
    path.write_text(json.dumps({"cases": {"private": {"rab_support_gbp": 1.6e9}}}), encoding="utf-8")
    cost, source = register_booked_cost("sizewell_c_rab", {"public_account": str(path), "key": "rab_support_gbp"})
    assert cost == pytest.approx(1.6e9) and source["case"] == "private"
    with pytest.raises(ValueError):
        register_booked_cost("sizewell_c_rab", {"public_account": str(path), "key": "rab_support_gbp", "case": "p2"})


def test_the_tariff_replaces_the_levy_only_where_the_register_books_it():
    from scripts.public_ownership.nationalised_tariff import RAB_LINE, recovery_rates, register_replaced_lines

    assert register_replaced_lines({}) == {}
    assert register_replaced_lines({"private_rab_support_gbp": 0.0}) == {}
    extra = register_replaced_lines({"private_rab_support_gbp": 1.0})
    assert list(extra) == [RAB_LINE]
    bill = pd.DataFrame({"case_id": "private", "component_id": ["direct_energy", RAB_LINE],
                         "annual_gbp_per_3100kwh_ex_vat": [300.0, 8.0]})
    conv = {"factor": 1.04, "mwh": 3.1, "final_demand_mwh": 6e8, "cfd_eligible_mwh": 5e8, "ro_eligible_mwh": 5e8}
    plain = recovery_rates(bill, conv)
    with_rab = recovery_rates(bill, conv, extra_replaced=extra)
    assert RAB_LINE not in plain["replaced_rates"]
    assert with_rab["replaced_rates"][RAB_LINE] == pytest.approx(plain["rule_rates"]["final_demand"])


def test_compensation_is_valued_on_the_bm_table_when_the_bill_names_one():
    from scripts.consumer_bill.extract_future_balancing_costs import compensation_strike_source

    overrides = [{"cfd_round": "HPC"}]
    rules = {"model_linked_rules": {"cfd": {"strike_table": "a.csv", "strike_overrides": overrides}}}
    assert compensation_strike_source(rules, "a.csv") == ("a.csv", overrides)
    rules["model_linked_rules"]["cfd"]["bm_strike_table"] = "resources/b.csv"
    path, got = compensation_strike_source(rules, "a.csv")
    assert path.endswith("b.csv") and got is None


# ── (b) the break-even calibration ───────────────────────────────────────────

def test_the_calibration_breaks_each_cohort_even_and_re_derives_to_itself():
    from scripts.market.cfd_strike_table import StrikeTable

    table = StrikeTable({("PAR7_2040", "onshore"): 50.0}, 2024, {}, {})
    register = pd.DataFrame({"case_id": "private", "asset_id": ["a", "b", "d"], "carrier": "wind_onshore",
                             "support_type": ["CfD", "CfD", "DPA"], "cfd_round": ["PAR7_2040", "PAR7_2040", "DPA_2040"],
                             "producer_rent_gbp": [-3.0e6, -1.0e6, -2.0e6], "cfd_paid_mwh": [1.0e5, 1.0e5, 0.0]})
    book = pd.DataFrame({"unit": ["d"], "scheme": "DPA", "dpa_carrier": "CCS Gas", "dpa_technology": [""],
                         "build_year": [2040], "dpa_apr_gbp_per_kw_year": [300.0], "capacity_mw": [100.0],
                         "dpa_expected_availability": [0.9], "live_share": [1.0]})
    frame = vk.derive_calibration(register, book, table, amortisation=vk.CONTRACT_TERM_WITH_TAIL,
                                  base_strikes={("PAR7_2040", "onshore"): 50.0},
                                  base_dpa={("CCS Gas", "", 2040): 300.0}, source="test").set_index("key")
    assert frame.loc["PAR7_2040_onshore", "calibrated_value"] == pytest.approx(50.0 + 4.0e6 / 2.0e5)
    assert frame.loc["DPA_2040_CCS Gas_", "calibrated_value"] == pytest.approx(300.0 + 2.0e6 / (100.0 * 1e3 * 0.9))
    # At break even the rent is zero and the value re-derives to itself.
    even = register.assign(producer_rent_gbp=0.0)
    again = vk.derive_calibration(even, book, table, amortisation=vk.CONTRACT_TERM_WITH_TAIL,
                                  base_strikes={("PAR7_2040", "onshore"): 50.0},
                                  base_dpa={("CCS Gas", "", 2040): 300.0}, source="test")
    assert (again["calibrated_value"] - again["value_in_use"]).abs().max() == 0.0


def test_a_stale_calibration_is_refused():
    with pytest.raises(vk.VintageCapitalError):
        vk.check_base(50.0, 51.0, "PAR7_2040_onshore")
    table = pd.DataFrame([{c: "" for c in vk.CALIBRATION_COLUMNS}])
    table["scheme"], table["basis"] = "CfD", vk.ECONOMIC_LIFE
    table["cfd_round"], table["technology"], table["base_value"], table["calibrated_value"] = "PAR7_2040", "onshore", 1.0, 2.0
    with pytest.raises(vk.VintageCapitalError):
        vk.calibrated_strikes(table, amortisation=vk.CONTRACT_TERM_WITH_TAIL)
