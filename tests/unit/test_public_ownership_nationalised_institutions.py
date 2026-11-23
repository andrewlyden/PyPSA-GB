"""Item N10: the nationalised bill under each dispatch institution.

Only four lines may move between institutions; the rest is capacity-driven and
must be carried across untouched and labelled as such. The measurement is taken
the same way from both years so that only dispatch differs.
"""

import json

import numpy as np
import pandas as pd
import pytest

from scripts.public_ownership.nationalised_institutions import (
    CARRIED_UNCHANGED_NOTE, compare, energy_table, measure_dispatch, measure_link_trade, substitute_lines,
)
from scripts.public_ownership.nationalised_tariff import (
    LINE_FIXED_OM, LINE_FORECAST_ERROR, LINE_FUEL, LINE_IC_RETRADING, LINE_IMPORTS, LINE_LEGACY_RENEWABLES,
    LINE_NEW_BUILD, LINE_REALTIME, LINE_VARIABLE_OM,
)

I1_LINES = {
    LINE_LEGACY_RENEWABLES: 3.9e9, LINE_NEW_BUILD: 10.06e9, LINE_FIXED_OM: 5.65e9,
    LINE_VARIABLE_OM: 0.6e9, LINE_FUEL: 4.53e9, LINE_IMPORTS: 2.28e9,
    LINE_REALTIME: 0.0, LINE_IC_RETRADING: 1.855e9,
}
MARKET = {"fuel_carbon_gbp": 5.0e9, "fuel_gbp": 3.6e9, "carbon_gbp": 1.4e9,
          "variable_om_gbp": 0.5e9, "interconnector_cost_gbp": 0.7e9,
          "energy_mwh": 300.0e6, "energy_by_carrier_mwh": {"CCGT": 20.0e6, "wind_offshore": 150.0e6}}
CENTRAL = {"fuel_carbon_gbp": 4.99e9, "fuel_gbp": 3.57e9, "carbon_gbp": 1.42e9,
           "variable_om_gbp": 0.5e9, "interconnector_cost_gbp": -0.72e9,
           "energy_mwh": 300.0e6, "energy_by_carrier_mwh": {"CCGT": 19.0e6, "wind_offshore": 152.0e6}}
I1_SUMMARY = {
    "conversion": {"factor": 1.038, "final_demand_mwh": 318.29e6, "mwh": 3.1, "vat_rate": 0.05},
    "revenue_requirement_gbp": I1_LINES,
    "bill": {"ex_vat_gbp": 817.10, "private_ex_vat_gbp": 913.63, "private_total_gbp_inc_vat": 959.31,
             "delta_inc_vat_gbp": -101.36},
}


class TestSubstitution:
    def test_only_the_four_lines_move(self):
        lines, audit = substitute_lines(I1_LINES, MARKET, CENTRAL, realtime_correction_gbp=-31.8e6)
        moved = {k for k in lines if not np.isclose(lines[k], I1_LINES.get(k, 0.0))}
        # No charge: the correction is already inside the physical lines, so the
        # real-time line does not move (review 26 Sep 2026).
        assert moved == {LINE_FUEL, LINE_IMPORTS, LINE_IC_RETRADING}
        # Variable O&M is identical in both years here, so it is offered but does not move.
        assert lines[LINE_VARIABLE_OM] == pytest.approx(I1_LINES[LINE_VARIABLE_OM])
        for untouched in (LINE_LEGACY_RENEWABLES, LINE_NEW_BUILD, LINE_FIXED_OM):
            assert lines[untouched] == I1_LINES[untouched]

    def test_each_moved_line_is_the_i1_level_plus_the_measured_delta(self):
        lines, audit = substitute_lines(I1_LINES, MARKET, CENTRAL, realtime_correction_gbp=-31.8e6)
        assert lines[LINE_FUEL] == pytest.approx(4.53e9 + (4.99e9 - 5.0e9))
        assert lines[LINE_IMPORTS] == pytest.approx(2.28e9 + (-0.72e9 - 0.7e9))
        assert audit["measured_deltas_gbp"][LINE_FUEL] == pytest.approx(-0.01e9)
        # The day-ahead position disappears, so its unwinding cost does too.
        assert lines[LINE_IC_RETRADING] == 0.0
        assert audit["interconnector_retrading_removed_gbp"] == pytest.approx(1.855e9)
        assert lines[LINE_REALTIME] == 0.0
        assert audit["realtime_correction_in_physical_lines_gbp"] == pytest.approx(-31.8e6)

    def test_the_resource_change_is_reported_as_a_share(self):
        _, audit = substitute_lines(I1_LINES, MARKET, CENTRAL, realtime_correction_gbp=0.0)
        # Review B13: fuel and variable O&M only; carbon is an Exchequer receipt.
        assert audit["generator_resource_cost_market_gbp"] == pytest.approx(4.1e9)
        assert audit["generator_resource_cost_central_gbp"] == pytest.approx(4.07e9)
        assert audit["generator_resource_cost_change_share"] == pytest.approx(-0.03e9 / 4.1e9)

    def test_carbon_is_split_out_of_the_fuel_line_as_an_exchequer_receipt(self):
        lines, audit = substitute_lines(I1_LINES, MARKET, CENTRAL, realtime_correction_gbp=0.0)
        # The bill line still pays fuel and carbon; the audit splits its delta.
        assert lines[LINE_FUEL] == pytest.approx(4.53e9 - 0.01e9)
        assert audit["fuel_delta_gbp"] == pytest.approx(-0.03e9)
        assert audit["carbon_delta_gbp"] == pytest.approx(0.02e9)
        _, summary = compare(I1_SUMMARY, MARKET, CENTRAL, realtime_correction_gbp=0.0)
        carbon_hh = 0.02e9 * 1.038 / 318.29e6 * 3.1 * 1.05
        assert summary["exchequer_carbon_receipt_change_inc_vat_gbp"] == pytest.approx(carbon_hh)
        assert summary["institution_change_net_of_carbon_receipt_inc_vat_gbp"] == pytest.approx(
            summary["institution_change_inc_vat_gbp"] - carbon_hh)
        bad = {**CENTRAL, "carbon_gbp": 2.0e9}
        with pytest.raises(ValueError, match="does not add up"):
            substitute_lines(I1_LINES, MARKET, bad, realtime_correction_gbp=0.0)

    def test_foreign_half_rent_is_remeasured_and_the_acquisition_carried(self):
        from scripts.public_ownership.nationalised_tariff import LINE_IC_ACQUISITION, LINE_IC_FOREIGN_RENT
        i1 = {**I1_LINES, LINE_IC_FOREIGN_RENT: 300.0e6, LINE_IC_ACQUISITION: 90.0e6}
        lines, audit = substitute_lines(i1, MARKET, CENTRAL, realtime_correction_gbp=0.0,
                                        central_foreign_rent_gbp=220.0e6)
        assert lines[LINE_IC_FOREIGN_RENT] == pytest.approx(220.0e6)
        assert lines[LINE_IC_ACQUISITION] == pytest.approx(90.0e6)  # a price paid once: capacity-driven
        assert audit["measured_deltas_gbp"][LINE_IC_FOREIGN_RENT] == pytest.approx(-80.0e6)
        with pytest.raises(ValueError, match="foreign-half rent"):
            substitute_lines(i1, MARKET, CENTRAL, realtime_correction_gbp=0.0)


class TestComparison:
    @pytest.fixture(scope="class")
    def built(self):
        return compare(I1_SUMMARY, MARKET, CENTRAL, realtime_correction_gbp=-31.8e6)

    def test_every_line_carries_its_basis(self, built):
        frame, _ = built
        by_line = frame.set_index("component_id")["basis"]
        assert by_line[LINE_FUEL] == "measured from both solved years"
        assert by_line[LINE_IC_RETRADING] == "institution line"
        assert by_line[LINE_NEW_BUILD] == CARRIED_UNCHANGED_NOTE

    def test_the_bill_moves_by_the_sum_of_the_line_changes(self, built):
        frame, summary = built
        assert summary["i3_ex_vat_gbp"] == pytest.approx(
            summary["i1_ex_vat_gbp"] + frame["change_gbp_per_3100kwh_ex_vat"].sum())
        assert summary["institution_change_ex_vat_gbp"] == pytest.approx(
            frame["change_gbp_per_3100kwh_ex_vat"].sum())
        # VAT is applied once, on each side.
        assert summary["i3_inc_vat_gbp"] == pytest.approx(summary["i3_ex_vat_gbp"] * 1.05)
        # Both deltas are computed from the bill quantities, so each closes exactly
        # whatever rounding the incoming tariff summary carried.
        for case in ("i1", "i3"):
            assert summary[f"{case}_delta_vs_private_inc_vat_gbp"] == pytest.approx(
                summary[f"{case}_inc_vat_gbp"] - summary["private_inc_vat_gbp"], abs=1e-9)
        assert summary["i1_delta_reported_by_tariff_inc_vat_gbp"] == -101.36

    def test_household_conversion_matches_the_i1_denominators(self, built):
        frame, _ = built
        row = frame.set_index("component_id").loc[LINE_NEW_BUILD]
        expected = 10.06e9 * 1.038 / 318.29e6 * 3.1
        assert row["i1_gbp_per_3100kwh_ex_vat"] == pytest.approx(expected)
        assert row["change_gbp_per_3100kwh_ex_vat"] == 0.0

    def test_summary_is_json_serialisable(self, built):
        json.dumps(built[1], default=float)


def test_energy_table_keeps_only_carriers_that_matter():
    table = energy_table(MARKET, CENTRAL, threshold_mwh=1.0e6).set_index("carrier")
    assert table.loc["CCGT", "change_twh"] == pytest.approx(-1.0)
    assert table.loc["wind_offshore", "change_twh"] == pytest.approx(2.0)


def test_chp_is_measured_net_of_its_heat_credit():
    from scripts.public_ownership.nationalised_institutions import heat_credit_rates, heat_credit_settings
    rates = pd.DataFrame({"carrier": ["CHP", "CCGT"], "fuel_cost_gbp_per_mwh": [80.0, 66.0],
                          "carbon_cost_gbp_per_mwh": [60.0, 50.0], "efficiency_used": [0.4, 0.55]},
                         index=["chp1", "ccgt1"])
    credit = heat_credit_rates(rates, {"CHP": {"heat_to_power": 1.1, "boiler_efficiency": 0.85}})
    factor = 1.1 * 0.4 / 0.85
    assert credit.loc["chp1", "heat_credit_fuel_gbp_per_mwh"] == pytest.approx(80.0 * factor)
    assert credit.loc["chp1", "heat_credit_carbon_gbp_per_mwh"] == pytest.approx(60.0 * factor)
    assert (credit.loc["ccgt1"] == 0.0).all()
    overlay = {"marginal_costs": {"byproduct_adjustments": {
        "CHP": {"mode": "heat_credit", "heat_to_power": 1.2, "boiler_efficiency": 0.9},
        "waste_to_energy": {"mode": "gate_fee_covers_fuel_and_carbon"}}}}
    # The brownfield overlay (load_layer("method", "brownfield")) is read after the config.
    assert heat_credit_settings({}, (overlay,)) == {"CHP": {"heat_to_power": 1.2, "boiler_efficiency": 0.9}}
    assert heat_credit_settings(overlay) == {"CHP": {"heat_to_power": 1.2, "boiler_efficiency": 0.9}}
    with pytest.raises(ValueError, match="efficiency_used"):
        heat_credit_rates(rates.drop(columns="efficiency_used"), {"CHP": {"heat_to_power": 1.1, "boiler_efficiency": 0.85}})


@pytest.mark.parametrize("measure", [lambda s: measure_dispatch(s, pd.DataFrame()), measure_link_trade])
def test_measurement_refuses_an_unsolved_month(measure):
    with pytest.raises(FileNotFoundError, match="must be solved"):
        measure(["not_a_scenario_that_exists"])


class TestForecastErrorChargeAndAllocation:
    """Audit 26 Sep 2026: I3 carries the borrowed FTI charge; lines convert at the tariff's own rates."""

    def test_the_borrowed_charge_is_an_i3_line_that_raises_the_i3_bill(self):
        _, without = compare(I1_SUMMARY, MARKET, CENTRAL, realtime_correction_gbp=0.0)
        frame, with_charge = compare(I1_SUMMARY, MARKET, CENTRAL, realtime_correction_gbp=0.0,
                                     forecast_error_charge_gbp=1.036e9)
        row = frame.set_index("component_id").loc[LINE_FORECAST_ERROR]
        assert row["i1_system_gbp"] == 0.0 and row["i3_system_gbp"] == pytest.approx(1.036e9)
        assert row["basis"].startswith("borrowed from FTI")
        # Recovered over final demand like the balancing line it stands beside.
        assert row["change_gbp_per_3100kwh_ex_vat"] == pytest.approx(1.036e9 * 1.038 / 318.29e6 * 3.1)
        assert with_charge["institution_change_ex_vat_gbp"] == pytest.approx(
            without["institution_change_ex_vat_gbp"] + row["change_gbp_per_3100kwh_ex_vat"])
        # The gross change is what the model measures itself, before the borrowed charge.
        assert with_charge["institution_change_gross_inc_vat_gbp"] == pytest.approx(
            without["institution_change_inc_vat_gbp"])

    def test_the_charge_replaces_the_correction_already_in_the_physical_lines(self):
        # User decision 26 Sep 2026: replace, not add on top. The measured lines are
        # physical dispatch, so the correction is already in them (review 26 Sep 2026):
        # replacing takes it back out, and without a charge nothing is added again.
        frame, charged = compare(I1_SUMMARY, MARKET, CENTRAL, realtime_correction_gbp=-52.9e6,
                                 forecast_error_charge_gbp=1.036e9)
        assert frame.set_index("component_id").loc[LINE_REALTIME, "i3_system_gbp"] == pytest.approx(52.9e6)
        assert charged["audit"]["realtime_correction_replaced_by_charge_gbp"] == pytest.approx(-52.9e6)
        frame, own = compare(I1_SUMMARY, MARKET, CENTRAL, realtime_correction_gbp=-52.9e6)
        assert frame.set_index("component_id").loc[LINE_REALTIME, "i3_system_gbp"] == 0.0
        _, zero = compare(I1_SUMMARY, MARKET, CENTRAL, realtime_correction_gbp=0.0)
        assert own["institution_change_inc_vat_gbp"] == pytest.approx(zero["institution_change_inc_vat_gbp"])
        # The gross change is the model's own measurement on physical dispatch.
        assert charged["institution_change_gross_inc_vat_gbp"] == pytest.approx(own["institution_change_inc_vat_gbp"])
        # The settlement's flag decides, even when the charge in a window is zero.
        frame, flagged = compare(I1_SUMMARY, MARKET, CENTRAL, realtime_correction_gbp=-52.9e6,
                                 replace_realtime_correction=True)
        assert frame.set_index("component_id").loc[LINE_REALTIME, "i3_system_gbp"] == pytest.approx(52.9e6)

    def test_lines_convert_at_the_tariff_allocation_not_flat(self):
        flat = 1.038 / 318.29e6 * 3.1
        allocation = {"new_line_rates": {LINE_FUEL: flat * 1.04, LINE_IMPORTS: flat * 1.04,
                                         LINE_VARIABLE_OM: flat * 1.04, LINE_NEW_BUILD: flat * 1.04,
                                         LINE_REALTIME: flat, LINE_IC_RETRADING: flat, LINE_FORECAST_ERROR: flat}}
        frame, summary = compare({**I1_SUMMARY, "allocation": allocation}, MARKET, CENTRAL,
                                 realtime_correction_gbp=0.0)
        by_line = frame.set_index("component_id")
        assert by_line.loc[LINE_FUEL, "change_gbp_per_3100kwh_ex_vat"] == pytest.approx(-0.01e9 * flat * 1.04)
        assert by_line.loc[LINE_IC_RETRADING, "change_gbp_per_3100kwh_ex_vat"] == pytest.approx(-1.855e9 * flat)
        assert summary["institution_change_ex_vat_gbp"] == pytest.approx(by_line["change_gbp_per_3100kwh_ex_vat"].sum())


def test_realtime_correction_is_taken_on_the_moved_lines_basis():
    """Review 27 Sep 2026: physical less schedule, fuel+carbon, variable O&M and cross-border trade only."""
    from scripts.public_ownership.nationalised_institutions import realtime_correction_on_line_basis
    schedule = {**CENTRAL, "fuel_carbon_gbp": 4.95e9, "variable_om_gbp": 0.49e9, "interconnector_cost_gbp": -0.70e9,
                "internal_cost_gbp": 9.9e9}  # internal links are in no moved line and must not count
    assert realtime_correction_on_line_basis(CENTRAL, schedule) == pytest.approx(0.04e9 + 0.01e9 - 0.02e9)
    with pytest.raises(ValueError, match="stage"):
        measure_dispatch(["x"], pd.DataFrame(), stage="outturn")


def test_a_stage_1_only_month_reads_its_link_tables_from_its_stage_1_network(tmp_path, monkeypatch):
    # 13 Oct 2026: the calibrated months take Stage 1 from s1_zf, which has no balancing network.
    import scripts.public_ownership.nationalised_institutions as ni
    monkeypatch.setattr(ni, "MARKET_DIR", tmp_path)
    (tmp_path / "parent_wholesale.nc").write_bytes(b"")
    assert ni._stage_1_network_path("parent") == tmp_path / "parent_wholesale.nc"
    (tmp_path / "parent_balancing.nc").write_bytes(b"")
    assert ni._stage_1_network_path("parent") == tmp_path / "parent_balancing.nc"
