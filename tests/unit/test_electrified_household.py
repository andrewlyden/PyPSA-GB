"""The electrified household beside the 3,100 kWh benchmark (user, 13 Nov 2026)."""

import pytest

from scripts.public_ownership import electrified_household as eh

BASIS = {"archetype": "heat_pump_ev_2050", "kwh": 6200.0, "benchmark_kwh": 3100.0, "private_standing_ex_vat": 300.0,
         "shares": {"tnuos": 1.0, "supplier_costs_and_margin": 0.3, "duos": 0.3},
         "inherit": {"tnuos_public_financing_adjustment": "tnuos", "interconnector_cap_and_floor_pass_through": "tnuos"}}


def test_the_basis_reads_the_headline_archetype_and_its_shares():
    summary = {"diagnostics": {"household_archetypes": {
        "headline_archetype": "heat_pump_ev_2050",
        "bills": [{"archetype": "benchmark", "annual_consumption_kwh": 3100, "standing_charge_gbp_ex_vat": 300.0},
                  {"archetype": "heat_pump_ev_2050", "annual_consumption_kwh": 6200, "standing_charge_gbp_ex_vat": 300.0}],
        "charge_basis": {"standing_share": {"tnuos": 1.0}}}}}
    basis = eh.archetype_basis(summary, {"charge_basis": {"inherit": {"x": "tnuos"}}})
    assert basis["kwh"] == 6200.0 and basis["benchmark_kwh"] == 3100.0
    assert basis["private_standing_ex_vat"] == 300.0
    assert eh.share("x", basis) == 1.0 and eh.share("cfd", basis) == 0.0
    assert eh.archetype_basis({"diagnostics": {"household_archetypes": {}}}, {}) is None


def test_a_standing_line_does_not_scale_and_a_unit_line_does():
    # private: 1,000 ex VAT, 300 standing -> electrified 300 + 700 x 2 = 1,700
    assert eh.bill_at(1000.0, 300.0, BASIS) == pytest.approx(1700.0)
    # a GBP 10 cut on a per-kWh line saves GBP 20 at twice the use ...
    assert eh.bill_at(990.0, 300.0, BASIS) - 1700.0 == pytest.approx(-20.0)
    # ... a GBP 10 cut on the standing charge saves GBP 10
    assert eh.bill_at(990.0, 290.0, BASIS) - 1700.0 == pytest.approx(-10.0)


def test_the_tariff_standing_charge_follows_its_line_changes():
    bill = {"replaced_lines_gbp_per_3100kwh_ex_vat": {"cfd": 100.0, "duos": 10.0},
            "dropped_lines_gbp_per_3100kwh_ex_vat": {"interconnector_cap_and_floor_pass_through": -5.0},
            "new_lines_gbp_per_3100kwh_ex_vat": {"nationalised_legacy_acquisition_renewables_nuclear": 150.0},
            "network_adjustments_gbp_per_3100kwh_ex_vat": {"tnuos_public_financing_adjustment": -4.0},
            "supplier_margin_adjustment_gbp_per_3100kwh_ex_vat": -2.0}
    # 300 - 0.3 x 10 - 1.0 x (-5) + 0 - 4 + 0.3 x (-2) = 297.4
    assert eh.tariff_standing(bill, BASIS) == pytest.approx(297.4)


def test_the_columns_against_private_and_both_tax_keys():
    rows = [{"case_id": "private", "bill_ex_vat": 1000.0, "electrified_standing_charge_gbp_ex_vat": 300.0,
             "taxpayer_share_gbp_per_household": 0.0, "taxpayer_share_legal_incidence_gbp_per_household": 0.0},
            {"case_id": "s2", "bill_ex_vat": 950.0, "electrified_standing_charge_gbp_ex_vat": 300.0,
             "taxpayer_share_gbp_per_household": 10.0, "taxpayer_share_legal_incidence_gbp_per_household": 7.0}]
    eh.add_columns(rows, BASIS, vat_rate=0.05)
    s2 = rows[1]
    assert s2["electrified_bill_delta_vs_private_inc_vat"] == pytest.approx(-100.0 * 1.05)
    assert s2["electrified_taxpayer_share_gbp_per_household"] == pytest.approx(20.0)
    assert s2["electrified_household_total_delta_vs_private_gbp_inc_vat"] == pytest.approx(-105.0 + 20.0)
    assert s2["electrified_household_total_legal_incidence_delta_vs_private_gbp_inc_vat"] == pytest.approx(-105.0 + 7.0)
    assert rows[0]["electrified_bill_delta_vs_private_inc_vat"] == 0.0


def test_a_row_without_a_standing_charge_is_refused():
    with pytest.raises(eh.ElectrifiedHouseholdError):
        eh.add_columns([{"case_id": "private", "bill_ex_vat": 1.0, "electrified_standing_charge_gbp_ex_vat": None}],
                       BASIS, vat_rate=0.05)


def test_the_high_demand_household_has_its_own_columns_on_the_same_standing_charge():
    rows = [{"case_id": "private", "bill_ex_vat": 1000.0, "electrified_standing_charge_gbp_ex_vat": 300.0,
             "taxpayer_share_gbp_per_household": 0.0},
            {"case_id": "s2", "bill_ex_vat": 950.0, "electrified_standing_charge_gbp_ex_vat": 300.0,
             "taxpayer_share_gbp_per_household": 10.0}]
    eh.add_columns(rows, BASIS, vat_rate=0.0)
    eh.add_columns(rows, dict(BASIS, kwh=9300.0, archetype="high"), vat_rate=0.0, prefix=eh.HIGH_PREFIX)
    assert rows[1]["electrified_bill_delta_vs_private_inc_vat"] == pytest.approx(-100.0)
    assert rows[1]["electrified_high_bill_delta_vs_private_inc_vat"] == pytest.approx(-150.0)
    assert rows[1]["electrified_high_taxpayer_share_gbp_per_household"] == pytest.approx(30.0)
    assert rows[1]["electrified_high_household_total_legal_incidence_delta_vs_private_gbp_inc_vat"] is None
    assert set(eh.COLUMNS) >= {"electrified_high_bill_inc_vat", "electrified_bill_inc_vat"}
    summary = {"diagnostics": {"household_archetypes": {"headline_archetype": "a",
        "bills": [{"archetype": "benchmark", "annual_consumption_kwh": 3100, "standing_charge_gbp_ex_vat": 3.0},
                  {"archetype": "a", "annual_consumption_kwh": 5000, "standing_charge_gbp_ex_vat": 3.0}],
        "charge_basis": {"standing_share": {"tnuos": 1.0}}}}}
    assert eh.archetype_basis(summary, {}, key="high_demand_archetype") is None
