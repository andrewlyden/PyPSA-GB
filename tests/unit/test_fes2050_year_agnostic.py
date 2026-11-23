"""The finance and bill layer takes its year from the run.

Two properties per change: the CP30 year reads exactly what it read before (the
2030 parity the task is accepted on, proved end to end by the parity harness),
and another model year reads its own year's values rather than 2030's, or
refuses where no value exists yet (the inputs tasks Y11 to Y13 must supply).
"""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.public_ownership import hpc_schedule as hs
from scripts.public_ownership.model_year import (
    DEFAULT_MODEL_YEAR, calendar_months, cm_calendar_revenue_column, cm_delivery_year_revenue_columns,
    cm_rate_columns, fiscal_year_label, frame_model_year, group_model_year, recovery_model_year, year_offsets,
)

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "resources" / "public_ownership" / "workbook_v5_assumptions.yaml"
CONV = {"factor": 1.038374717833, "final_demand_mwh": 318.29e6, "mwh": 3.1, "vat_rate": 0.05}


def test_the_default_year_is_the_cp30_year():
    assert DEFAULT_MODEL_YEAR == 2030
    assert group_model_year({}) == 2030 and group_model_year({"model_year": 2050}) == 2050
    assert recovery_model_year(None) == 2030 and recovery_model_year({"forecast_year": 2050}) == 2050
    with pytest.raises(ValueError, match="integer year"):
        group_model_year({"model_year": 2050.5})


def test_a_register_has_one_model_year():
    assert frame_model_year(pd.DataFrame({"model_year": [2050, 2050]})) == 2050
    assert frame_model_year(pd.DataFrame({"x": [1]})) == 2030
    with pytest.raises(ValueError, match="mixes model years"):
        frame_model_year(pd.DataFrame({"model_year": [2030, 2050]}))


def test_year_names_reproduce_the_cp30_builder_columns():
    # scripts/market/build_capacity_market.py writes these names for calendar 2030.
    assert cm_calendar_revenue_column(2030) == "calendar_2030_revenue_gbp"
    assert cm_delivery_year_revenue_columns(2030) == ("revenue_2029_30_gbp", "revenue_2030_31_gbp")
    assert cm_delivery_year_revenue_columns(2050) == ("revenue_2049_50_gbp", "revenue_2050_51_gbp")
    assert fiscal_year_label(2030, "/") == "2030/31" and fiscal_year_label(2099) == "2099_00"
    columns = ["carrier", "derated_mw_2029_30", "connection_mw_2029_30", "revenue_2029_30_gbp",
               "revenue_2030_31_gbp", "t1_2029_30_mw", "calendar_2030_revenue_gbp", "gbp_per_kw_connection_year"]
    assert cm_rate_columns(columns) == ["revenue_2029_30_gbp", "revenue_2030_31_gbp", "calendar_2030_revenue_gbp",
                                        "gbp_per_kw_connection_year"]
    assert cm_rate_columns(["calendar_2050_revenue_gbp", "revenue_2049_50_gbp"]) == [
        "calendar_2050_revenue_gbp", "revenue_2049_50_gbp"]
    assert year_offsets(2030, (0, 7)) == (2030, 2037)


def test_capacity_market_rates_are_rebased_for_any_year():
    from scripts.public_ownership.calculate_financing_scenarios import capacity_market_rates_in_price_year

    rates = pd.DataFrame({"carrier": ["CCGT"], "calendar_2050_revenue_gbp": [2.0], "revenue_2049_50_gbp": [4.0],
                          "gbp_per_kw_connection_year": [8.0], "derated_mw_2049_50": [10.0]})
    out = capacity_market_rates_in_price_year(rates, 2024, (2024, 2025, 2.0))
    assert out.loc[0, ["calendar_2050_revenue_gbp", "revenue_2049_50_gbp", "gbp_per_kw_connection_year"]].tolist() == [
        1.0, 2.0, 4.0]
    assert out.loc[0, "derated_mw_2049_50"] == 10.0  # capacity, not money


@pytest.mark.skipif(not SNAPSHOT.exists(), reason="needs the workbook snapshot")
def test_the_yield_shift_is_the_model_years_column_of_the_workbook_path():
    from scripts.public_ownership.finance import snapshot_model_year, with_model_year, yield_shift, yield_shift_cell

    snapshot = yaml.safe_load(SNAPSHOT.read_text(encoding="utf-8"))
    for analogue in ("Sc2", "Sc3", "Sc4"):
        for year in (2025, 2030, 2050):
            # Schema 4 carries a year-named block for every year of the workbook's gilt paths.
            named = snapshot[f"sovereign_yield_shift_{year}"][analogue]["B"]
            assert yield_shift_cell(snapshot, analogue, "B", year) is named
    with pytest.raises(ValueError, match="gilt paths run 2025 to 2050"):
        yield_shift_cell(snapshot, "Sc3", "B", 2052)
    with pytest.raises(ValueError, match="assumption C is blank"):
        yield_shift_cell(snapshot, "Sc3", "C", 2030)
    settings = {"sovereign_yield": "B", "public_return": "public_rate"}
    assert snapshot_model_year(snapshot) == 2030
    assert yield_shift(snapshot, settings) == snapshot["sovereign_yield_shift_2030"]["Sc3"]["B"]["value"]
    assert yield_shift(snapshot, {**settings, "sovereign_yield": "A"}) == 0.0
    later = with_model_year(snapshot, 2050)
    assert yield_shift(later, settings) == snapshot["sovereign_yield_shift_2050"]["Sc3"]["B"]["value"]
    assert yield_shift(later, settings) != yield_shift(snapshot, settings)
    assert "_model_year" not in snapshot  # a view, the snapshot itself is untouched


def test_hpc_months_follow_the_prices_year():
    assert list(map(str, hs.MONTHS)) == [str(m) for m in pd.period_range("2030-01", "2030-12", freq="M")]
    assert str(calendar_months(2050)[0]) == "2050-01"
    schedule = {"edf_schedule": {"unit_1_first_power": "2030-06-01", "ramp_months": 6, "unit_1_mw": 1630,
                                 "availability_at_full_power": 0.9}}
    prices_2050 = {str(m): 50.0 for m in calendar_months(2050)}
    shortfall = hs.schedule_shortfall(schedule, 18.0e6, prices_2050)
    # Long after first power the unit runs all year: its months are 2050's, not 2030's.
    assert list(shortfall["edf_mwh_by_month"]) == list(prices_2050)
    assert shortfall["edf_mwh"] == pytest.approx(1630 * 0.9 * 8760)
    with pytest.raises(ValueError, match="one model year"):
        hs.prices_model_year({"2030-12": 1.0, "2031-01": 1.0})


def test_multi_year_years_count_from_the_model_year():
    from scripts.public_ownership.nationalised_tariff import (
        LINE_LEGACY_RENEWABLES, LINE_LEGACY_THERMAL, MULTI_YEAR_OFFSETS, headline_years, multi_year,
    )

    assert year_offsets(2030, MULTI_YEAR_OFFSETS) == (2030, 2032, 2035, 2037, 2040, 2045, 2050, 2060)
    profile = pd.DataFrame({"scope_group": ["valued"] * 2, "carrier": ["wind_offshore"] * 2, "support_type": [""] * 2,
                            "stream": ["net_operating_cash"] * 2, "year": [2050, 2060], "gbp": [10.0e6, 5.0e6]})
    lines = {LINE_LEGACY_RENEWABLES: 1.0e6, LINE_LEGACY_THERMAL: 0.0}
    frame = multi_year(profile, lines, CONV, term_years=30, model_year=2050)
    assert frame["year"].iloc[:8].tolist() == [2050, 2052, 2055, 2057, 2060, 2065, 2070, 2080]
    out = headline_years(frame.assign(year=frame["year"].astype(str)), {"private_total_gbp_inc_vat": 1000.0}, CONV,
                         None, model_year=2050)
    assert "2050 snapshot" in out and "2030 snapshot" not in out
    assert "peak year" not in out["note"]  # the 2030 finding is not asserted of another year
    cp30 = headline_years(frame.assign(year=frame["year"].astype(str)), {"private_total_gbp_inc_vat": 1000.0}, CONV,
                          None)
    assert cp30["note"].startswith("2030 is the peak year")


def test_network_account_refuses_a_year_no_price_control_covers():
    from scripts.public_ownership.network_finance_account import (
        build_account, distribution_rav, explicit_rav_key, settings,
    )

    blocks = {
        "distribution": {"price_basis_key": "2020/21", "closing_rav_gbp_m_2027_28": 69027.0,
                         "net_additions_gbp_m_2027_28": 6076.7, "depreciation_gbp_m_2027_28": 4000.4,
                         "tax_allowance_gbp_m_period_average": 735.5, "vanilla_wacc_cpih_real": 0.0393},
        "transmission": {"price_basis_key": "2023/24", "total_rav_gbp_m": 54210.2, "wacc_semi_nominal": 0.0564,
                         "notional_gearing": 0.55},
    }
    assert explicit_rav_key() == "distribution_rav_2030_gbp" and explicit_rav_key(2050) == "distribution_rav_2050_gbp"
    cfg = settings({"distribution_roll_forward": "ed2_net_growth"})
    assert "distribution_rav_2030_gbp" in cfg
    # 2030's roll-forward rule is refused for 2050, where it would return a 2030 asset base.
    with pytest.raises(ValueError, match="after RIIO-ED3"):
        distribution_rav(blocks, cfg, 2050)
    explicit = settings({"distribution_roll_forward": "explicit", "distribution_rav_2050_gbp": 1.5e11}, 2050)
    assert distribution_rav(blocks, explicit, 2050) == (1.5e11, "explicit configured value")
    with pytest.raises(ValueError, match="distribution_rav_2050_gbp must be positive"):
        settings({"distribution_roll_forward": "explicit", "distribution_rav_2030_gbp": 9.0e10}, 2050)
    with pytest.raises(ValueError, match="outside RIIO-3"):
        build_account(blocks, 0.0275, cfg, CONV, model_year=2050)
    _, summary = build_account(blocks, 0.0275, cfg, CONV)
    assert summary["transmission"]["rav_basis"] == "RIIO-3 period-average RAV; 2030 is inside the determined period"


def test_price_control_coverage_is_stated_for_the_model_year():
    from scripts.consumer_bill.extract_network_price_control_building_blocks import coverage

    assert coverage(2030)["transmission_covered"] and not coverage(2050)["transmission_covered"]
    assert "never the 2050 answer" in coverage(2050)["distribution_note"]


def test_legacy_horizon_must_follow_the_model_year():
    from scripts.public_ownership.legacy_valuation import settings_from_config

    assert settings_from_config({"horizon_year": 2045})["horizon_year"] == 2045
    with pytest.raises(ValueError, match="after the model year"):
        settings_from_config({"horizon_year": 2045}, model_year=2050)


def test_ro_bill_period_is_the_recovery_files_year():
    from scripts.consumer_bill.build_ro_eligible_fleet import (
        OBLIGATION_YEAR, OBLIGATION_YEAR_START, PERIOD_END, PERIOD_START, bill_period,
    )

    assert (OBLIGATION_YEAR, PERIOD_START, PERIOD_END, OBLIGATION_YEAR_START) == (
        "calendar 2030", pd.Timestamp("2030-01-01"), pd.Timestamp("2031-01-01"), pd.Timestamp("2030-04-01"))
    label, start, end, april = bill_period(2050)
    assert (label, start, end, april) == ("calendar 2050", pd.Timestamp("2050-01-01"), pd.Timestamp("2051-01-01"),
                                          pd.Timestamp("2050-04-01"))


def test_bill_year_keys_follow_the_bill_calendar_year():
    from scripts.consumer_bill.project_future_bill import tnuos_household_share
    from scripts.consumer_bill.refresh_recovery_demand import check_capacity_market_pin

    tnuos = {"derivation": {"domestic_residual_gbp_per_year": 50.0, "demand_residual_revenue_gbp_m_2050_51": 5000.0}}
    assert tnuos_household_share({"benchmark": {"bill_calendar_year": 2050}, "sourced_externals": {"tnuos": tnuos}}) \
        == pytest.approx(50.0 / 5.0e9)
    assumptions = {"benchmark": {"bill_calendar_year": 2050},
                   "sourced_externals": {"capacity_market": {"derivation": {"agreement_cost_gbp": 1.0e9}}}}
    assert check_capacity_market_pin(assumptions, {"calendar_2050_cost_gbp": 1.0e9})["agrees"]
    with pytest.raises(ValueError, match="calendar_2050_cost_gbp"):
        check_capacity_market_pin(assumptions, {"calendar_2030_cost_gbp": 1.0e9})


def test_workbook_feedback_names_its_columns_for_the_model_year():
    # F29: workbook v2.0 writes a gilt-shift block per year; the model year's column applies.
    from scripts.public_ownership.finance import with_model_year, yield_shift, yield_shift_cell

    snapshot = {"sovereign_yield_shift_2030": {"Sc4": {"B": {"value": 0.0020}}},
                "sovereign_yield_shift_2050": {"Sc4": {"B": {"value": 0.0035}}}}
    assert yield_shift_cell(snapshot, "Sc4", "B", 2030)["value"] == 0.0020
    assert yield_shift_cell(snapshot, "Sc4", "B", 2050)["value"] == 0.0035
    settings = {"new_build_generation": "public", "public_return": "public_rate", "legacy_fleet": "acquire_all",
                "sovereign_yield": "B"}
    assert yield_shift(with_model_year(snapshot, 2050), settings) == pytest.approx(0.0035)
    with pytest.raises(ValueError, match="sovereign_yield_shift_2040"):
        yield_shift_cell(snapshot, "Sc4", "B", 2040)


def test_a_run_group_may_omit_its_model_year():
    from scripts.public_ownership.reference_runs import _group_year

    assert _group_year({}) == 2030 and _group_year({"model_year": 2050}) == 2050
    with pytest.raises(ValueError, match="integer year"):
        _group_year({"model_year": "2050"})
