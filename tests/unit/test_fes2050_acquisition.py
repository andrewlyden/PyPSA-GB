"""FES 2050 plan, task Y12 (decision 2): the 2030 acquisition carried to 2050.

The state buys the legacy fleet in 2030; in 2050 it still pays that annuity (to 2059),
whatever survives.  Surviving plant costs its operating cost only, retired plant is
replaced at vintage cost and the state rate, and every ledger closes.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.public_ownership import acquisition_carry_forward as acf
from scripts.public_ownership.finance import NATIONALISED_CASE, capital_recovery_factor
from scripts.public_ownership.legacy_valuation import settings_from_config, value_legacy_fleet
from scripts.public_ownership.nationalised_tariff import (
    LINE_IC_ACQUISITION, LINE_IC_FOREIGN_RENT, LINE_LEGACY_RENEWABLES, LINE_LEGACY_THERMAL, VALUE, build_bill,
    multi_year, reconcile, revenue_requirement,
)

RATE = 0.027450980392157
CRF30 = capital_recovery_factor(RATE, 30.0)
CONV = {"factor": 1.04, "final_demand_mwh": 600.0e6, "cfd_eligible_mwh": 560.0e6, "ro_eligible_mwh": 570.0e6,
        "mwh": 3.1, "vat_rate": 0.05}


def _case_2030(term=30.0, financing="tariff", exchequer=None, case_id=NATIONALISED_CASE):
    valued, thermal = 80.0e9, 20.0e9
    rr = {LINE_LEGACY_RENEWABLES: valued * CRF30, LINE_LEGACY_THERMAL: thermal * CRF30,
          acf.LINE_CANCELLED_RESIDUAL: 0.0, LINE_IC_ACQUISITION: 13.0e9 * CRF30, acf.LINE_TRANSACTION_COSTS: 1.0e7}
    diag = {"acquisition_price_gbp": valued + thermal, "acquisition_price_renewables_nuclear_gbp": valued,
            "acquisition_price_thermal_storage_gbp": thermal, "interconnector_acquisition_price_gbp": 13.0e9,
            "interconnector_acquisition_terms": {"public_rate": RATE}}
    if exchequer:
        diag["acquisition_annuity_exchequer_funded_by_line_gbp"] = {k: rr[k] for k in exchequer}
        for k in exchequer:
            rr[k] = 0.0
    return {"case_id": case_id, "model_year": 2030, "levers": {"legacy_fleet": "acquire_all"},
            "acquisition_terms": {"purchase_price_basis": "market_value", "thermal_basis": "book_value",
                                  "term_years": term, "acquisition_financing": financing},
            "revenue_requirement_gbp": rr, "diagnostics": diag}


def _valuation():
    return pd.DataFrame([
        # retired offshore wind (2039), replaced in 2039
        {"asset_id": "off", "carrier": "wind_offshore", "scope_group": "valued", "capacity_mw": 1000.0,
         "retirement_year": 2039.0, "retirement_method": "commissioning_plus_life", "cfd_contract_end_year": 2032.0},
        # Hinkley Point C: operating in 2050
        {"asset_id": "Hinkley Point C::legacy", "carrier": "nuclear", "scope_group": "valued", "capacity_mw": 3260.0,
         "retirement_year": 2065.0, "retirement_method": "named_override", "cfd_contract_end_year": 2065.0},
        # onshore that an extended life keeps to 2051
        {"asset_id": "on", "carrier": "wind_onshore", "scope_group": "valued", "capacity_mw": 500.0,
         "retirement_year": 2046.0, "retirement_method": "commissioning_plus_life", "cfd_contract_end_year": np.nan},
        # retired gas: not replaced like for like
        {"asset_id": "ccgt", "carrier": "CCGT", "scope_group": "thermal", "capacity_mw": 800.0,
         "retirement_year": 2032.0, "retirement_method": "commissioning_plus_life", "cfd_contract_end_year": np.nan},
        # foreign plant: never acquired
        {"asset_id": "fr", "carrier": "EU_market", "scope_group": "excluded", "capacity_mw": 2000.0,
         "retirement_year": 2040.0, "retirement_method": "undated_default_remaining_years", "cfd_contract_end_year": np.nan},
    ]).assign(public_real_rate=RATE, horizon_year=2070)


SNAPSHOT = {"construction_risk_premium_by_rating": {"L-M": {"value": 0.005}, "M": {"value": 0.0075}},
            "technologies": [{"technology": "Offshore wind", "risk_rating": "M"},
                             {"technology": "Onshore wind", "risk_rating": "L-M"},
                             {"technology": "Large-scale nuclear", "risk_rating": "L-M"}]}


class TestCarriedAnnuity:
    def test_the_2030_annuity_is_paid_in_2050_whatever_survives(self):
        carried = acf.carried_acquisition(_case_2030(), model_year=2050)
        assert carried["payable_in_model_year"] and carried["last_payment_year"] == 2059
        assert carried["lines_model_year_gbp"][LINE_LEGACY_RENEWABLES] == pytest.approx(80.0e9 * CRF30)
        assert carried["total_model_year_gbp"] == pytest.approx((80 + 20 + 13) * 1e9 * CRF30 + 1.0e7)
        assert carried["annuity_factor_implied"] == pytest.approx(CRF30)

    def test_a_fifteen_year_term_has_ended_by_2050(self):
        carried = acf.carried_acquisition(_case_2030(term=15.0), model_year=2050)
        assert not carried["payable_in_model_year"] and carried["last_payment_year"] == 2044
        assert carried["total_model_year_gbp"] == 0.0
        assert acf.carried_acquisition(_case_2030(), model_year=2060)["total_model_year_gbp"] == 0.0

    def test_an_exchequer_funded_annuity_is_still_owed_and_still_off_the_tariff(self):
        carried = acf.carried_acquisition(
            _case_2030(financing="exchequer", exchequer=(LINE_LEGACY_RENEWABLES, LINE_LEGACY_THERMAL)), model_year=2050)
        assert carried["lines_model_year_gbp"][LINE_LEGACY_RENEWABLES] == 0.0
        assert carried["gross_lines_model_year_gbp"][LINE_LEGACY_RENEWABLES] == pytest.approx(80.0e9 * CRF30)
        assert carried["on_tariff_model_year_gbp"] < carried["total_model_year_gbp"]

    def test_only_an_acquisition_year_case_can_be_carried(self):
        case = dict(_case_2030(), model_year=2050)
        with pytest.raises(ValueError, match="must come from the 2030 tariff"):
            acf.carried_acquisition(case, model_year=2050)


class TestSurvivalAndReplacement:
    def test_survival_follows_the_valuation_rules_and_the_lives_lever(self):
        groups = acf.ACQUIRED_GROUPS["acquire_all"]
        survival = acf.survival_schedule(_valuation(), [2031, 2050], groups=groups).set_index(["year", "carrier"])
        assert survival.at[(2031, "wind_offshore"), "cfd_live_mw"] == 1000.0
        assert survival.at[(2050, "wind_offshore"), "operating_mw"] == 0.0
        assert survival.at[(2050, "nuclear"), "operating_mw"] == 3260.0
        assert (2050, "EU_market") not in survival.index
        extended = acf.survival_schedule(_valuation(), [2050], groups=groups, asset_lives="extended",
                                         extended={"extra_years": 5, "carriers": ["wind_onshore"]})
        assert extended.set_index("carrier").at["wind_onshore", "operating_mw"] == 500.0

    def test_retired_plant_is_rebuilt_at_its_vintage_cost_and_the_state_rate(self):
        from scripts.public_ownership.vintage_costs import costs_for

        rebuilt = acf.replacement_build(_valuation(), 2050, RATE, groups=acf.ACQUIRED_GROUPS["acquire_all"],
                                        snapshot=SNAPSHOT).set_index("carrier")
        assert set(rebuilt.index) == {"wind_offshore", "wind_onshore"}  # gas is not replaced like for like
        off = rebuilt.loc["wind_offshore"]
        cost = costs_for("Offshore wind", 2039)
        # F29 (26 Oct 2026): the state's cash rate, with no construction premium.
        assert off["build_year"] == 2039 and off["state_rate"] == pytest.approx(RATE)
        assert off["capex_gbp"] == pytest.approx(1000.0e3 * cost["capex_power_gbp_per_kw"])
        assert off["annual_capital_gbp"] == pytest.approx(
            off["capex_gbp"] * capital_recovery_factor(RATE, float(cost["economic_life_years"])))

    def test_the_ledger_closes_and_refuses_a_broken_capacity_identity(self):
        groups = acf.ACQUIRED_GROUPS["acquire_all"]
        carried = acf.carried_acquisition(_case_2030(), model_year=2050)
        survival = acf.survival_schedule(_valuation(), [2050], groups=groups)
        rebuilt = acf.replacement_build(_valuation(), 2050, RATE, groups=groups, snapshot=SNAPSHOT)
        ledger = acf.acquisition_ledger(carried, survival, rebuilt, 2050)
        capacity = ledger["capacity_mw"]
        assert capacity["acquired"] == pytest.approx(capacity["operating"] + capacity["retired"])
        assert capacity["retired"] == pytest.approx(capacity["replaced"] + capacity["retired_not_replaced_like_for_like"])
        assert ledger["parts_gbp"]["surviving_acquired_plant_capital_gbp"] == 0.0
        assert ledger["annuity_gbp"] == pytest.approx(carried["total_model_year_gbp"])
        with pytest.raises(ValueError, match="does not close"):
            acf.acquisition_ledger(carried, survival, rebuilt.assign(capacity_mw=rebuilt["capacity_mw"] * 0.5), 2050)


def _row(case, asset_id, carrier, vintage, *, owned, scope=None, capital=0.0, fixed=0.0, fuel=0.0, market=0.0,
         support=0.0, price=np.nan):
    return {
        "case_id": case, "asset_id": asset_id, "carrier": carrier, "finance_vintage": vintage, "state_owned": owned,
        "legacy_scope_group": scope, "annual_capital_recovery_gbp": capital, "annual_fixed_om_gbp": fixed,
        "annual_variable_om_gbp": 0.0, "annual_required_revenue_gbp": capital + fixed, "fuel_carbon_gbp": fuel,
        "market_revenue_gbp": market, "support_revenue_gbp": support, "acquisition_price_gbp": price,
        "capacity_mw": 100.0, "dispatch_mwh": 1.0e6, "model_year": 2050, "wholesale_revenue_gbp": market,
        "balancing_settlement_gbp": 0.0, "cfd_support_gbp": 0.0, "roc_support_gbp": 0.0,
    }


def _register_2050():
    """A 2050 register: a surviving legacy nuclear unit and state new build."""
    return pd.DataFrame([
        _row("private", "hpc", "nuclear", "legacy", owned=False, capital=900.0e6, fixed=200.0e6, fuel=60.0e6,
             market=1500.0e6),
        _row("private", "new_off", "wind_offshore", "new_build", owned=False, capital=800.0e6, fixed=100.0e6,
             market=700.0e6, support=300.0e6),
        # The register would re-price the survivor at a 2050 purchase: the carried case must set it aside.
        _row(NATIONALISED_CASE, "hpc", "nuclear", "legacy", owned=True, scope="valued", capital=400.0e6,
             fixed=200.0e6, fuel=60.0e6, market=1500.0e6, price=7.0e9),
        _row(NATIONALISED_CASE, "new_off", "wind_offshore", "new_build", owned=True, capital=500.0e6, fixed=100.0e6,
             market=700.0e6, support=300.0e6),
    ])


def _private_bill():
    values = {"direct_energy": 150.0, "cfd": 60.0, "bsuos_constraints": 30.0, "capacity_market": 10.0,
              "tnuos": 90.0, "duos": 120.0, "supplier_costs_and_margin": 160.0}
    return pd.DataFrame([{
        "case_id": "private", "scenario": "HT50", "modelled_year": 2050, "component_id": k, "stack_group": "x",
        "category": "x", "component": k, VALUE: v, "source_class": "x", "method": "m", "source": "s", "currency": "GBP",
        "reference_gbp_per_3100kwh_ex_vat": np.nan} for k, v in values.items()])


class TestTariff2050:
    @pytest.fixture(scope="class")
    def carried(self):
        return acf.carried_acquisition(_case_2030(), model_year=2050)

    def test_surviving_plant_costs_operating_cost_only_and_the_annuity_is_its_own_line(self, carried):
        lines, diag = revenue_requirement(_register_2050(), carried_acquisition=carried, interconnector_trade_gbp=0.0,
                                          interconnector={LINE_IC_FOREIGN_RENT: 1.0e8, LINE_IC_ACQUISITION: 9.9e8},
                                          interconnector_detail={})
        assert lines[LINE_LEGACY_RENEWABLES] == pytest.approx(80.0e9 * CRF30)
        assert lines[LINE_LEGACY_THERMAL] == pytest.approx(20.0e9 * CRF30)
        # The GB halves were bought in 2030 too: the carried annuity, not a 2050 price.
        assert lines[LINE_IC_ACQUISITION] == pytest.approx(13.0e9 * CRF30)
        assert diag["register_legacy_capital_set_aside_gbp"] == pytest.approx(400.0e6)
        assert diag["acquisition_price_gbp"] == pytest.approx(100.0e9)
        assert diag["acquisition_carried_from_year"] == 2030 and diag["acquisition_carried_last_payment_year"] == 2059
        # The survivor's operating costs stay; new build keeps its capital.
        assert lines["nationalised_fixed_om"] == pytest.approx(300.0e6)
        assert lines["nationalised_new_build_capital"] == pytest.approx(500.0e6)

    def test_both_ledgers_close_on_the_2050_fixture(self, carried):
        register, private = _register_2050(), _private_bill()
        lines, diag = revenue_requirement(register, carried_acquisition=carried)
        components, summary = build_bill(private, lines, CONV)
        recon, recon_summary = reconcile(private, components, lines, diag, CONV)
        assert abs(recon_summary["unexplained_gbp"]) <= recon_summary["register_ledger_tolerance_gbp"]
        assert recon_summary["attribution_gbp"]["legacy acquisition annuity, renewables and nuclear"] == pytest.approx(
            80.0e9 * CRF30)
        bill_steps = recon.loc[recon["ledger"].eq("bill"), "gbp_per_3100kwh_ex_vat"].sum()
        assert bill_steps == pytest.approx(summary["delta_ex_vat_gbp"])

    def test_the_multi_year_view_charges_the_annuity_to_2059_only(self, carried):
        profile = pd.DataFrame([{"scope_group": "valued", "carrier": "nuclear", "support_type": "CfD",
                                 "stream": "net_operating_cash", "year": y, "gbp": 1.0e9} for y in range(2050, 2081)])
        lines = {LINE_LEGACY_RENEWABLES: 80.0e9 * CRF30, LINE_LEGACY_THERMAL: 20.0e9 * CRF30}
        years = multi_year(profile, lines, CONV, term_years=30.0, model_year=2050, acquisition_year=2030,
                           horizon_year=2080).set_index("year")
        assert years.at[2050, "acquisition_annuity_gbp"] == pytest.approx(100.0e9 * CRF30)
        assert years.at[2060, "acquisition_annuity_gbp"] == 0.0 and years.at[2080, "acquisition_annuity_gbp"] == 0.0
        # 15-year average: ten of fifteen years carry the annuity.
        assert years.at["15-year average", "acquisition_annuity_gbp"] == pytest.approx(100.0e9 * CRF30 * 10 / 15)
        with pytest.raises(ValueError, match="silent zeros"):
            multi_year(profile, lines, CONV, term_years=30.0, model_year=2050, acquisition_year=2030, horizon_year=2070)


class TestLegacyHorizon:
    @staticmethod
    def _register(year):
        return pd.DataFrame([{
            "case_id": "private", "asset_id": "hydro", "component_type": "Generator", "carrier": "large_hydro",
            "finance_vintage": "legacy", "support_type": "merchant", "cfd_round": None, "cfd_award_year": None,
            "capacity_mw": 100.0, "commissioning_year": 2040, "market_revenue_gbp": 10.0, "cfd_support_gbp": 0.0,
            "cfd_curtailment_compensation_gbp": 0.0, "roc_support_gbp": 0.0, "fuel_carbon_gbp": 0.0,
            "annual_fixed_om_gbp": 1.0, "wacc": 0.07, "model_year": year}])

    STATIONS = pd.DataFrame({"Technology": ["Onshore wind"], "Station DNC in kW": ["1,000"],
                             "Effective date": ["01/04/2012"]})

    def test_a_2050_profile_reaches_2080_and_says_so(self):
        settings = settings_from_config({"horizon_year": 2070}, 2050)
        _, profile, summary = value_legacy_fleet(self._register(2050), RATE, settings, self.STATIONS)
        assert summary["horizon_year"] == 2080 and summary["horizon_year_configured"] == 2070
        assert "no silent zero years" in summary["horizon_extension"]
        assert int(profile["year"].max()) == 2080

    def test_the_cp30_horizon_is_unchanged(self):
        settings = settings_from_config({"horizon_year": 2070}, 2030)
        _, _, summary = value_legacy_fleet(self._register(2030), RATE, settings, self.STATIONS)
        assert summary["horizon_year"] == 2070 and "horizon_extension" not in summary
