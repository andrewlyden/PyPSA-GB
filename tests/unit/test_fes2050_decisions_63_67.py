"""FES 2050 decisions 63 to 67 and the repowering bound's own valuation (3 Oct 2026).

Decision 63: a battery revenue top-up (support_lines_2050), paid to the row's owner, replaced in the
tariff. Decision 65: the 2050 BSUoS lines' services part is the bottom-up ancillary table, so the
services the register and outside providers are paid equal what the lines hold. Limitation 8: the
0.5 repowering bound values its legacy rows on its own share. Decision 66: merchant rows on a level
annuity over their economic life. Decision 67: the batteries' declared sub-hourly revenue. Light
synthetic fixtures only.
"""

import json

import numpy as np
import pandas as pd
import pytest

A2050 = "data/market/consumer_bill_benchmarks/2050_pypsa_informed_assumptions.json"
ANCILLARY = "data/public_ownership/financing/ancillary_services_2050.csv"


# ── decision 63: the battery top-up ───────────────────────────────────────────

def test_battery_top_up_is_the_private_shortfall_on_battery_rows_only():
    from scripts.public_ownership import support_lines_2050 as sl

    private = pd.DataFrame({"case_id": "private", "asset_id": ["B1", "B2", "B3", "L1", "G1"],
                            "carrier": ["Battery", "Battery", "Battery", "LAES", "CCS Gas"],
                            "producer_rent_gbp": [-50.0, 20.0, np.nan, -80.0, -10.0]})
    top = sl.battery_top_up_by_row(private, ["Battery"], ldes_carriers=["LAES", "CAES"])
    assert top.to_dict() == {"B1": 50.0, "B2": 0.0, "B3": 0.0, "L1": 0.0, "G1": 0.0}
    with pytest.raises(sl.SupportLinesError, match="LDES"):
        sl.battery_top_up_by_row(private, ["Battery", "LAES"], ldes_carriers=["LAES"])
    # F19 decision 5 (6 Oct 2026) adds the basis; a vintaged (2050) register keeps the private basis.
    parsed = {"carriers": ["Battery"], "basis": "owner_cost", "owner_cost_in_vintaged_registers": False}
    assert sl.settings({"battery_top_up": {"carriers": ["Battery"]}}) == {"battery_top_up": parsed}
    assert sl.settings({"battery_top_up": {}}) == {"battery_top_up": parsed}
    assert sl.owner_cost_top_up({"battery_top_up": parsed}, vintaged=True) is None
    assert sl.settings({"battery_top_up": {"enabled": False}}) is None
    assert sl.TOP_UP_LINES[sl.COLUMN_BATTERY_TOP_UP] == "battery_revenue_top_up"


def test_a_state_held_battery_keeps_the_margin_of_its_lower_rate():
    from scripts.public_ownership import support_lines_2050 as sl

    # Private: required 150 at the private rate, revenue 100 -> rent -50, top-up 50.  P2: the state owns it
    # and its required revenue at the public rate is 120 -> rent -20 before the top-up; after it, +30,
    # all of it the state's surplus (returned to bills under the base incidence).
    top = pd.Series({"B1": 50.0})
    reg = pd.DataFrame({"case_id": ["private", "p2"], "asset_id": ["B1", "B1"], "producer_rent_gbp": [-50.0, -20.0],
                        "public_surplus_before_recycling_gbp": [0.0, -20.0], "ownership_share": [0.0, 1.0]})
    booked = sl.book_support(reg, {sl.COLUMN_BATTERY_TOP_UP: top})
    assert booked["producer_rent_gbp"].tolist() == [0.0, 30.0]
    assert booked["public_surplus_before_recycling_gbp"].tolist() == [0.0, 30.0]
    assert booked[sl.COLUMN_BATTERY_TOP_UP].tolist() == [50.0, 50.0]


def test_the_top_up_is_sized_after_every_other_line_and_a_reference_holds_them():
    from scripts.public_ownership import support_lines_2050 as sl
    from scripts.public_ownership.calculate_financing_scenarios import support_lines_by_asset

    merged = pd.DataFrame({"asset_id": ["B1", "W1"], "carrier": ["Battery", "waste_to_energy"],
                           "capacity_mw": [100.0, 10.0], "dispatch_mwh": [0.0, 1000.0]})
    private = pd.DataFrame({"case_id": "private", "asset_id": ["B1", "W1"], "carrier": ["Battery", "waste_to_energy"],
                            "producer_rent_gbp": [-80.0, -5.0], "public_surplus_before_recycling_gbp": [0.0, 0.0],
                            "ownership_share": [0.0, 0.0]})
    support = {"battery_top_up": {"carriers": ["Battery"]}}
    out = support_lines_by_asset(merged, private, support, book=lambda f: f, ldes_carriers=["LAES"])
    assert out[sl.COLUMN_BATTERY_TOP_UP].to_dict() == {"B1": 80.0, "W1": 0.0}
    # A balancing group: the market group's per-asset amounts, whatever its own private rows say.
    reference = pd.DataFrame({"asset_id": ["B1", "W1"], sl.COLUMN_BATTERY_TOP_UP: [70.0, 0.0]})
    held = support_lines_by_asset(merged, private, support, book=lambda f: f, reference=reference)
    assert held[sl.COLUMN_BATTERY_TOP_UP].to_dict() == {"B1": 70.0, "W1": 0.0}
    with pytest.raises(sl.SupportLinesError, match="no row"):
        support_lines_by_asset(merged, private, support, book=lambda f: f, reference=reference.iloc[:1])
    assert support_lines_by_asset(merged, private, None, book=lambda f: f) == {}


def test_the_tariff_replaces_the_battery_line_only_when_the_register_pays_it():
    from scripts.public_ownership.nationalised_tariff import BATTERY_TOP_UP_LINE, register_replaced_lines

    assert register_replaced_lines({}) == {}
    assert BATTERY_TOP_UP_LINE in register_replaced_lines({"private_battery_top_up_gbp": 1.0})
    assert register_replaced_lines({"private_battery_top_up_gbp": 0.0}) == {}


def test_the_2050_bill_reads_the_battery_top_up_from_the_public_account():
    a = json.loads(open(A2050, encoding="utf-8").read())
    line = a["sourced_externals"]["battery_revenue_top_up"]
    assert line["derivation"]["cost_from"]["key"] == "battery_revenue_top_up_gbp"
    assert line["derivation"]["cost_from"]["case"] == "private"
    assert line["derivation"]["recovery_denominator"] == "final_demand_mwh" and not line.get("loss_uplift")


# ── decision 65: the BSUoS lines on the bottom-up services ──────────────────

def test_the_bsuos_lines_hold_exactly_the_services_paid():
    from scripts.public_ownership import support_lines_2050 as sl

    table = sl.load_ancillary(ANCILLARY)
    paid = sl.ancillary_by_provider(table, "central")
    held = sl.services_held_by_line(table, "central")
    assert sum(held.values()) == pytest.approx(float(paid["gbp"].sum()), abs=1e-6)
    a = json.loads(open(A2050, encoding="utf-8").read())
    for line, amount in held.items():
        block = a["sourced_externals"][line]
        assert block["derivation"]["cost_gbp"] == pytest.approx(amount, abs=1e-6), line
        assert block["derivation"]["ancillary_case"] == "central"
        low, high = sl.services_held_by_line(table, "low")[line], sl.services_held_by_line(table, "high")[line]
        assert block["derivation"]["sensitivity_cost_gbp"] == pytest.approx({"ancillary_low": low, "ancillary_high": high})
        assert not block.get("loss_uplift") and block["period_basis"] == "extrapolated_declared"
    # The register's support line is on the same case as the bill's central.
    from scripts.public_ownership.study_config import load_studies

    po = load_studies("config", validate=False).configs["FES25_HT_2050"]["public_ownership"]
    assert po["support_lines"]["ancillary_services"]["case"] == "central"
    other = a["sourced_externals"]["bsuos_other_services"]["derivation"]
    assert other["demand_scaled_volume"] == "final_demand_mwh"          # review 2: the rate's own base
    # internal 595.6 m plus the categories less the reserve trades, over the FY2024/25 volume
    fy = other["fy2024_25"]
    assert other["demand_scaled_gbp_per_mwh"] == pytest.approx(
        (fy["internal_and_incentive_gbp"] + fy["trade_categories_gbp"] - fy["reserve_trades_in_ancillary_table_gbp"])
        / fy["chargeable_volume_mwh"])


def test_the_projector_adds_the_demand_scaled_part_exactly():
    from scripts.consumer_bill.project_future_bill import resolve_sourced_externals

    block = {"unit": "gbp_per_household_year", "derivation": {
        "cost_gbp": 900e6, "recovery_denominator": "final_demand_mwh", "demand_scaled_gbp_per_mwh": 2.5,
        "demand_scaled_volume": "gsp_chargeable_volume_mwh"}}
    recovery = {"final_demand_mwh": 600e6, "gsp_chargeable_volume_mwh": 660e6}
    out = resolve_sourced_externals({"x": block}, recovery, 3100)["x"]
    cost = 900e6 + 2.5 * 660e6
    assert out["derivation"]["cost_gbp_total"] == pytest.approx(cost)
    assert out["value"] == pytest.approx(cost / 600e6 * 3.1)
    bad = {"unit": "gbp_per_household_year", "derivation": {**block["derivation"], "demand_scaled_volume": "x_mwh"}}
    with pytest.raises(ValueError, match="recovery volume"):
        resolve_sourced_externals({"x": bad}, recovery, 3100)


def test_the_ancillary_low_and_high_are_a_named_bill_sensitivity():
    from scripts.consumer_bill.project_future_bill import levy_and_cm_sensitivities

    frame = pd.DataFrame({"component_id": ["bsuos_other_services"], "annual_gbp_per_3100kwh_ex_vat": [10.0]})
    rebasing = {"from_price_year": 2024, "to_price_year": 2025, "from_index": 100.0, "to_index": 104.0, "factor": 1.04}
    assumptions = {"benchmark": {"price_year": 2025}, "sourced_externals": {"bsuos_other_services": {
        "price_year": 2024, "rebasing": rebasing, "derivation": {
            "cost_gbp": 900e6, "recovery_denominator": "final_demand_mwh",
            "sensitivity_cost_gbp": {"ancillary_low": 450e6, "ancillary_high": 1800e6}}}}}
    out = levy_and_cm_sensitivities(frame, assumptions, {"final_demand_mwh": 600e6}, 3100)
    assert out["bsuos_other_services_ancillary_low"] == pytest.approx(-450e6 / 600e6 * 3.1 * 1.04)
    assert out["bsuos_other_services_ancillary_high"] == pytest.approx(900e6 / 600e6 * 3.1 * 1.04)


# ── limitation 8: the repowering bound's own share ──────────────────────────

def test_a_group_with_its_own_repowering_rule_values_on_it():
    from scripts.public_ownership.legacy_valuation import _run_group_of, group_repowering_settings

    settings = {"repowering_share_by_carrier": {"wind_onshore": 0.3}, "max_life_years_by_carrier": {"wind_onshore": 35}}
    block = {"reference_run_groups": {
        "bound": {"fleet_vintages": {"repowering": {"Onshore wind": {"share": 0.5, "max_life_years": 35}}}},
        "central": {"fleet_vintages": {"repowering": {"Onshore wind": {"share": 0.3, "max_life_years": 35}}}},
        "cp30": {}}}
    assert group_repowering_settings(settings, block, "bound")["repowering_share_by_carrier"] == {"wind_onshore": 0.5}
    assert group_repowering_settings(settings, block, "central") == settings
    assert group_repowering_settings(settings, block, "cp30") == settings
    # F28 (d), 27 Oct 2026: with no central rule in the valuation, the group's own rule applies on its own
    assert group_repowering_settings({"x": 1}, block, "bound") == {
        "x": 1, "repowering_share_by_carrier": {"wind_onshore": 0.5}, "max_life_years_by_carrier": {"wind_onshore": 35.0}}
    assert group_repowering_settings({"x": 1}, block, "cp30") == {"x": 1}         # no group rule: unchanged
    assert _run_group_of("resources/public_ownership/fes_ht_2050_repowering_bound_asset_finance_register.parquet") \
        == "fes_ht_2050_repowering_bound"
    po = __import__("scripts.public_ownership.study_config", fromlist=["load_studies"]).load_studies(
        "config", validate=False).configs["FES25_HT_2050"]["public_ownership"]
    from scripts.public_ownership.legacy_valuation import settings_from_config

    base = settings_from_config(po["acquisition"]["valuation"], model_year=2050)
    bound = group_repowering_settings(base, po, "fes_ht_2050_repowering_bound")
    assert bound["repowering_share_by_carrier"]["wind_onshore"] == 0.5
    assert group_repowering_settings(base, po, "fes_ht_2050") == base


# ── decision 66: merchant rows on a level annuity ───────────────────────────

def test_merchant_rows_take_a_level_annuity_and_past_life_rows_are_sunk():
    from scripts.public_ownership.vintage_capital_2050 import merchant_level_rows

    merged = pd.DataFrame({"support_type": [None, None, "CfD", "RAB", None, None],
                           "economic_life_years": [15.0, 15.0, 30.0, 60.0, 15.0, 15.0],
                           "commissioning_year": [2036.0, 2030.0, 2040.0, 2036.0, np.nan, 2050.0],
                           "model_year": 2050})
    term = np.array([False, False, True, False, False, False])
    expired = np.zeros(6, dtype=bool)
    out = merchant_level_rows(merged, term, expired)
    assert out["level_row"].tolist() == [True, False, False, False, False, True]
    assert out["past_life_row"].tolist() == [False, True, False, False, False, False]


def test_the_level_annuity_is_capex_over_the_life_at_the_rows_own_rate():
    from scripts.public_ownership.calculate_financing_scenarios import _term_basis_capital
    from scripts.public_ownership.finance import capital_recovery_factor as crf

    n = 3
    terms = {"term_row": np.zeros(n, dtype=bool), "expired_row": np.zeros(n, dtype=bool),
             "term_years": np.full(n, np.nan), "live_share": np.ones(n), "tail_margin_gbp_per_kw_year": np.zeros(n),
             "life_years": np.full(n, np.nan), "level_row": np.array([True, False, False]),
             "past_life_row": np.array([False, True, False]), "level_life_years": np.array([15.0, 15.0, 15.0])}
    idx = np.arange(n)
    rate = np.array([0.08, 0.08, 0.035])
    capacity, energy = np.ones(n), np.full(n, 2.0)
    capex_power, capex_energy = np.full(n, 300.0), np.full(n, 100.0)
    remaining_basis = np.array([999.0, 999.0, 7.0])           # what the remaining-life rule had charged
    out = _term_basis_capital(terms, idx, rate, capacity, energy, capex_power, capex_energy, remaining_basis)
    assert out[0] == pytest.approx(500e3 * crf(0.08, 15))     # (300 + 2 x 100) GBP/kW x 1 MW
    assert out[1] == 0.0 and out[2] == 7.0                    # past its life: sunk; not merchant: unchanged
    # without the decision-66 keys (every other register) nothing changes
    plain = {k: v for k, v in terms.items() if k not in ("level_row", "past_life_row", "level_life_years")}
    assert _term_basis_capital(plain, idx, rate, capacity, energy, capex_power, capex_energy,
                               remaining_basis).tolist() == remaining_basis.tolist()


def test_the_basis_switch_is_validated():
    from scripts.public_ownership.vintage_capital_2050 import VintageCapitalError, basis_from_config

    assert basis_from_config(None)["merchant_capital"] == "remaining_life"
    with pytest.raises(VintageCapitalError):
        basis_from_config({"vintage_capital_2050": {"merchant_capital": "level"}})
    with pytest.raises(VintageCapitalError, match="term amortisation"):
        basis_from_config({"vintage_capital_2050": {"merchant_capital": "level_annuity"}})


# ── decision 67: the sub-hourly uplift ──────────────────────────────────────

def test_the_subhourly_uplift_is_income_and_the_top_up_is_sized_after_it():
    from scripts.public_ownership import support_lines_2050 as sl
    from scripts.public_ownership.calculate_financing_scenarios import support_lines_by_asset

    merged = pd.DataFrame({"asset_id": ["B1", "G1"], "carrier": ["Battery", "CCS Gas"], "capacity_mw": [100.0, 10.0],
                           "wholesale_revenue_gbp": [10.0, 50.0], "balancing_settlement_gbp": [40.0, 5.0]})
    private = pd.DataFrame({"case_id": "private", "asset_id": ["B1", "G1"], "carrier": ["Battery", "CCS Gas"],
                            "producer_rent_gbp": [-30.0, 0.0], "public_surplus_before_recycling_gbp": [0.0, 0.0],
                            "ownership_share": [0.0, 0.0]})
    raw = {"battery_top_up": {"carriers": ["Battery"]},
           "battery_subhourly_uplift": {"carriers": ["Battery"], "share_by_case": {"central": 0.2, "high": 0.4}}}
    support = sl.settings(raw)
    out = support_lines_by_asset(merged, private, support, book=lambda f: f)
    assert out[sl.COLUMN_SUBHOURLY].to_dict() == {"B1": 10.0, "G1": 0.0}           # 20% of 10 + 40
    assert out[sl.COLUMN_BATTERY_TOP_UP].to_dict() == {"B1": 20.0, "G1": 0.0}       # -30 + 10
    assert out[sl.COLUMN_BATTERY_TOP_UP_HIGH].to_dict() == {"B1": 10.0, "G1": 0.0}  # -30 + 20
    reg = private.assign(case_id="private")
    booked = sl.book_support(reg, {k: v for k, v in out.items() if not k.startswith("_")})
    # the uplift and the top-up are income; the high-case top-up is a memo
    assert booked["producer_rent_gbp"].tolist() == [0.0, 0.0]
    with pytest.raises(sl.SupportLinesError):
        sl.settings({"battery_subhourly_uplift": {"carriers": ["Battery"], "share_by_case": {"central": 0.2}}})


def test_the_tariff_credits_the_owned_batteries_subhourly_revenue():
    from scripts.public_ownership.nationalised_tariff import LINE_SUBHOURLY_INCOME, NEW_LINES, line_label

    assert LINE_SUBHOURLY_INCOME not in NEW_LINES and "decision 67" in line_label(LINE_SUBHOURLY_INCOME)


def test_a_register_booked_line_takes_its_high_case_from_the_public_account(tmp_path):
    from scripts.consumer_bill.project_future_bill import levy_and_cm_sensitivities

    account = tmp_path / "acct.json"
    account.write_text(json.dumps({"cases": {"private": {"battery_revenue_top_up_gbp": 370e6,
                                                         "battery_revenue_top_up_subhourly_high_gbp": 310e6}}}))
    rebasing = {"from_price_year": 2024, "to_price_year": 2025, "from_index": 100.0, "to_index": 104.0, "factor": 1.04}
    frame = pd.DataFrame({"component_id": ["battery_revenue_top_up"], "annual_gbp_per_3100kwh_ex_vat": [2.0]})
    assumptions = {"benchmark": {"price_year": 2025}, "sourced_externals": {"battery_revenue_top_up": {
        "price_year": 2024, "rebasing": rebasing, "derivation": {
            "cost_from": {"public_account": str(account), "case": "private", "key": "battery_revenue_top_up_gbp"},
            "recovery_denominator": "final_demand_mwh",
            "sensitivity_cost_from_keys": {"subhourly_high": "battery_revenue_top_up_subhourly_high_gbp"}}}}}
    out = levy_and_cm_sensitivities(frame, assumptions, {"final_demand_mwh": 600e6}, 3100)
    assert out["battery_revenue_top_up_subhourly_high"] == pytest.approx(-60e6 / 600e6 * 3.1 * 1.04)


# ── the battery items in 2030 (user, 3 Oct 2026): keyed, default off ─────────

def test_support_lines_are_off_unless_named_and_can_be_switched_off():
    from scripts.public_ownership import support_lines_2050 as sl

    assert sl.settings(None) is None and sl.settings({}) is None
    raw = {"ancillary_services": {"enabled": True, "table": "data/public_ownership/financing/ancillary_services_2030.csv",
                                  "case": "central"},
           "battery_subhourly_uplift": {"enabled": True, "central": 0.20, "high": 0.40},
           "battery_top_up": {"enabled": True}, "efw_gate_fee": {"enabled": False, "file": "x.json"}}
    out = sl.settings(raw)
    assert set(out) == {"ancillary_services", "battery_subhourly_uplift", "battery_top_up"}
    assert out["battery_subhourly_uplift"]["share"] == 0.20 and out["battery_subhourly_uplift"]["high_share"] == 0.40
    assert str(out["ancillary_services"]["file"]).endswith("ancillary_services_2030.csv")
    sl.require_vintaged(out, vintaged=False)                          # the battery items work on a 2030 register
    with pytest.raises(sl.SupportLinesError, match="vintaged"):
        sl.require_vintaged(sl.settings({"life_extension_top_up": {"technologies": {"Offshore wind": "expired"}}}),
                            vintaged=False)


def test_the_2030_table_pays_the_cp30_carriers():
    from scripts.public_ownership import support_lines_2050 as sl

    table = sl.load_ancillary("data/public_ownership/financing/ancillary_services_2030.csv")
    assets = pd.DataFrame({"asset_id": ["B", "C", "O", "G", "P"], "carrier": ["Battery", "CCGT", "OCGT", "gas_engine",
                                                                              "pumped_hydro"],
                           "capacity_mw": [1000.0, 1000.0, 500.0, 500.0, 1000.0]})
    out, report = sl.ancillary_by_row(assets, table)
    paid = sl.ancillary_by_provider(table)
    peaking = float(paid.loc[paid["provider_carrier"].eq("peaking_gas"), "gbp"].sum())
    assert out.loc["O", sl.COLUMN_ANCILLARY] == pytest.approx(peaking / 2)    # OCGT and gas engines pro rata
    assert out.loc["C", sl.COLUMN_ANCILLARY] > 0 and out.loc["B", sl.COLUMN_ANCILLARY] > 0
    # every service is its FY2024/25 outturn carried to 2030 on CP30's drivers (user decision, 6 Oct 2026),
    # the projection scripts/public_ownership/ancillary_services_2030.py writes (tests/unit/test_ancillary_services_2030.py)
    first = table.groupby("service").first()
    assert first.loc["voltage_bm_actions", "spend_2050_central_gbp_m_2024"] == first.loc["voltage_bm_actions", "outturn_spend_gbp_m"]
    assert first.loc["stability_network_services", "spend_2050_central_gbp_m_2024"] > first.loc["stability_network_services", "outturn_spend_gbp_m"]
    assert not set(table["provider_carrier"]) & {"CCS Gas", "H2_turbine", "CCS Biomass"}


def test_the_battery_items_book_on_a_non_vintaged_register():
    from scripts.public_ownership import support_lines_2050 as sl
    from scripts.public_ownership.calculate_financing_scenarios import support_lines_by_asset

    merged = pd.DataFrame({"asset_id": ["B1", "L1"], "carrier": ["Battery", "LAES"], "capacity_mw": [100.0, 50.0],
                           "wholesale_revenue_gbp": [50.0, 20.0], "balancing_settlement_gbp": [0.0, 0.0]})
    private = pd.DataFrame({"case_id": "private", "asset_id": ["B1", "L1"], "carrier": ["Battery", "LAES"],
                            "producer_rent_gbp": [-40.0, -10.0], "public_surplus_before_recycling_gbp": [0.0, 0.0],
                            "ownership_share": [0.0, 0.0]})
    support = sl.settings({"battery_top_up": {}, "battery_subhourly_uplift": {"central": 0.2, "high": 0.4}})
    out = support_lines_by_asset(merged, private, support, book=lambda f: f, ldes_carriers=["LAES"])
    assert out[sl.COLUMN_SUBHOURLY].to_dict() == {"B1": 10.0, "L1": 0.0}
    assert out[sl.COLUMN_BATTERY_TOP_UP].to_dict() == {"B1": 30.0, "L1": 0.0}


def test_the_support_lines_moved_out_of_the_capital_basis():
    from scripts.public_ownership.vintage_capital_2050 import VintageCapitalError, basis_from_config

    with pytest.raises(VintageCapitalError, match="moved to public_ownership.support_lines"):
        basis_from_config({"vintage_capital_2050": {"amortisation": "economic_life", "support_lines": {"x": 1}}})
