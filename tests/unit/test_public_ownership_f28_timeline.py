"""F28 (user decisions, 26 Oct 2026): the policy timeline, small-scale plant, per-asset repayment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.public_ownership.build_asset_finance_register import (
    SMALL_SCALE_EXCLUSION, exclude_small_scale, micro_capacity_by_carrier,
)
from scripts.public_ownership.calculate_financing_scenarios import acquisition_row_terms, crf_rows, plan_term_summary
from scripts.public_ownership.finance import capital_recovery_factor
from scripts.public_ownership.legislation_year import (
    STATUS_FLEET, STATUS_POST, STATUS_PRE, apply_legislation_year, load_lead_times,
)
from scripts.public_ownership.nationalised_tariff import (
    AnnuitySchedule, IC_ACQUISITION_DEFAULTS, interconnector_acquisition_settings, interconnector_remaining_years,
)
from scripts.public_ownership.acquisition_carry_forward import line_live_shares
from scripts.public_ownership.levers import TERM_REMAINING_LIFE

LEADS = load_lead_times()


def _row(asset_id, carrier, *, vintage="new_build", capacity=100.0, **extra):
    row = {"asset_id": asset_id, "carrier": carrier, "component_type": "Generator", "finance_vintage": vintage,
           "finance_eligible": True, "capacity_mw": capacity, "energy_capacity_mwh": 0.0, "dispatch_mwh": capacity * 1000,
           "support_type": None, "cfd_round": None, "cfd_award_year": np.nan, "contract_vintage": "not_applicable",
           "commissioning_year": np.nan, "development_status": None, "cohort_legacy_share": np.nan,
           "vintage_method": "asset_date"}
    row.update(extra)
    return row


def _apply(rows, year, **kwargs):
    register = pd.DataFrame(rows)
    return apply_legislation_year(register, legislated_year=year, policy_start_year=2025, model_year=2030,
                                  transfer_year=kwargs.pop("transfer_year", 2030), leads=LEADS, **kwargs)


def test_a_contract_is_judged_by_its_award_year():
    rows = [_row("ar7", "wind_offshore", support_type="CfD", cfd_round="AR7", cfd_award_year=2026.0,
                 contract_vintage="post_policy", commissioning_year=2029.0)]
    central = _apply(rows, 2028).set_index("asset_id")
    assert central.at["ar7", "finance_vintage"] == "legacy"
    assert central.at["ar7", "contract_vintage"] == "pre_policy"
    assert central.at["ar7", "legislation_status"] == STATUS_PRE
    bound = _apply(rows, 2025).set_index("asset_id")
    assert bound.at["ar7", "finance_vintage"] == "new_build"
    assert bound.at["ar7", "legislation_status"] == STATUS_POST


def test_merchant_plant_is_judged_by_its_investment_date():
    # A battery commissioned in 2029 was invested in 2028 (one year's construction): eligible at 2028.
    late = _apply([_row("b29", "Battery", commissioning_year=2029.0)], 2028).set_index("asset_id")
    assert late.at["b29", "finance_vintage"] == "new_build"
    # One commissioned in 2028 was invested in 2027: existing plant at the 2030 transfer.
    early = _apply([_row("b28", "Battery", commissioning_year=2028.0)], 2028).set_index("asset_id")
    assert early.at["b28", "finance_vintage"] == "legacy"


def test_undated_capacity_is_split_on_an_even_build_profile_and_settles_once():
    rows = [_row("solar::legacy", "solar_pv", vintage="legacy", capacity=40.0, cohort_legacy_share=0.4,
                 vintage_method="2024_to_model_capacity_delta_by_carrier"),
            _row("solar::new_build", "solar_pv", capacity=60.0, cohort_legacy_share=0.4,
                 vintage_method="2024_to_model_capacity_delta_by_carrier")]
    out = _apply(rows, 2028).set_index("asset_id")
    # 2025..2030 even; solar invested a year before commissioning, so 2029 and 2030 are eligible: 2/6.
    assert out.at["solar::new_build", "capacity_mw"] == pytest.approx(60.0 * 2 / 6)
    assert out.at["solar::pre_legislation", "capacity_mw"] == pytest.approx(60.0 * 4 / 6)
    assert out.at["solar::pre_legislation", "finance_vintage"] == "legacy"
    # Dated at the mean of its committed-before years (2025..2028), not the old fleet's median.
    assert out.at["solar::pre_legislation", "commissioning_year"] == pytest.approx(2026.5)
    # Every part of the asset settles on cohort_share, summing to one.
    assert out["cohort_share"].sum() == pytest.approx(1.0)
    assert out.at["solar::legacy", "cohort_share"] == pytest.approx(0.4)
    assert out["capacity_mw"].sum() == pytest.approx(100.0)
    assert out.at["solar::legacy", "legislation_status"] == STATUS_FLEET


def test_a_project_under_construction_in_the_snapshot_was_committed_before_any_law():
    out = _apply([_row("p", "wind_onshore", development_status="Under Construction",
                       vintage_method="development_status")], 2025).set_index("asset_id")
    assert out.at["p", "finance_vintage"] == "legacy"


def test_a_vintaged_build_after_the_transfer_committed_before_the_law_is_in_construction():
    rows = [_row("nuc::b2031", "nuclear", build_year=2031.0, cohort_share=1.0)]
    out = apply_legislation_year(pd.DataFrame(rows), legislated_year=2028, policy_start_year=2025, model_year=2050,
                                 transfer_year=2030, leads=LEADS, vintaged=True).set_index("asset_id")
    assert out.at["nuc::b2031", "finance_vintage"] == "in_construction"


def test_modelled_par7_nuclear_is_new_build_but_a_real_nuclear_contract_is_judged_by_its_award():
    par7 = _row("nuc::b2031", "nuclear", build_year=2031.0, cohort_share=1.0, support_type="CfD",
                cfd_round="PAR7_2031")
    hpc = _row("hpc", "nuclear", build_year=2031.0, cohort_share=1.0, support_type="CfD", cfd_round="HPC",
               cfd_award_year=2016.0)
    out = apply_legislation_year(pd.DataFrame([par7, hpc]), legislated_year=2028, policy_start_year=2025,
                                 model_year=2050, transfer_year=2030, leads=LEADS, vintaged=True).set_index("asset_id")
    assert out.at["nuc::b2031", "finance_vintage"] == "new_build"
    assert out.at["nuc::b2031", "legislation_status"] == STATUS_POST
    assert out.at["hpc", "finance_vintage"] == "in_construction"


def test_a_carrier_without_a_lead_time_is_refused():
    with pytest.raises(ValueError, match="no lead time"):
        _apply([_row("x", "unobtainium", commissioning_year=2029.0)], 2028)


def test_small_scale_rows_are_excluded_by_source_and_by_micro_share():
    register = pd.DataFrame([
        {"asset_id": "roof", "carrier": "solar_pv", "data_source": "CP30_small_scale_solar_gsp", "capacity_mw": 10.0,
         "dispatch_mwh": 1.0, "finance_eligible": True, "evidence_status": "resolved", "cohort_share": np.nan},
        {"asset_id": "fes1", "carrier": "solar_pv", "data_source": "FES", "capacity_mw": 60.0, "dispatch_mwh": 6.0,
         "finance_eligible": True, "evidence_status": "resolved", "cohort_share": 1.0},
        {"asset_id": "fes2", "carrier": "solar_pv", "data_source": "FES", "capacity_mw": 40.0, "dispatch_mwh": 4.0,
         "finance_eligible": True, "evidence_status": "resolved", "cohort_share": 1.0},
    ])
    out = exclude_small_scale(register, data_sources=["CP30_small_scale_solar_gsp"],
                              micro_mw_by_carrier={"solar_pv": 25.0}).set_index("asset_id")
    assert out.at["roof", "ownership_exclusion"] == SMALL_SCALE_EXCLUSION
    assert out.at["roof", "evidence_status"] == "excluded"
    assert out.at["fes1::small_scale", "capacity_mw"] == pytest.approx(15.0)
    assert out.at["fes1", "capacity_mw"] == pytest.approx(45.0)
    assert out.at["fes1::small_scale", "cohort_share"] + out.at["fes1", "cohort_share"] == pytest.approx(1.0)
    assert out["capacity_mw"].sum() == pytest.approx(110.0)
    assert not bool(out.at["fes2::small_scale", "finance_eligible"])
    with pytest.raises(ValueError, match="exceeds"):
        exclude_small_scale(register, micro_mw_by_carrier={"solar_pv": 200.0})


def test_micro_capacity_comes_from_the_es1_distributed_micro_supply_lines():
    lines = pd.DataFrame({"connection": ["Distributed - Micro", "Distributed - Micro", "Transmission"],
                          "role": ["supply", "storage", "supply"], "quantity": ["capacity_mw"] * 3,
                          "carrier": ["solar_pv", "Battery", "solar_pv"], "value_std": [33098.6, 7413.7, 100.0]})
    assert micro_capacity_by_carrier(lines) == {"solar_pv": 33098.6}


def test_each_asset_is_repaid_over_its_remaining_life():
    terms = acquisition_row_terms(TERM_REMAINING_LIFE, [2045, 2031, 2030], [2030, 2030, 2030])
    assert terms.tolist() == [15.0, 1.0, 1.0]
    assert acquisition_row_terms("30", [2045], [2030]).tolist() == [30.0]
    with pytest.raises(ValueError, match="retirement year"):
        acquisition_row_terms(TERM_REMAINING_LIFE, [np.nan], [2030])
    plan = pd.DataFrame({"acquisition_price_gbp": [100.0, 300.0], "acquisition_term_years": [10.0, 20.0]})
    plan["acquisition_annuity_factor"] = crf_rows(0.03, plan["acquisition_term_years"])
    summary = plan_term_summary(plan, 0.03, TERM_REMAINING_LIFE)
    assert summary["term_years"] == pytest.approx(17.5)
    assert summary["annuity_factor"] * 400.0 == pytest.approx(100.0 * capital_recovery_factor(0.03, 10)
                                                               + 300.0 * capital_recovery_factor(0.03, 20))
    assert summary["last_payment_offset_years"] == 20.0


def test_the_annuity_schedule_runs_off_asset_by_asset_and_keeps_the_present_value():
    schedule = AnnuitySchedule({10.0: 30.0, 20.0: 70.0})
    assert schedule.live(0) == pytest.approx(1.0)
    assert schedule.live(10) == pytest.approx(0.7)
    assert schedule.live(20) == 0.0
    fixed = AnnuitySchedule.fixed(30.0)
    assert fixed.live(29) == 1.0 and fixed.live(30) == 0.0
    # Bought in the model year, a pound of price is worth (1 + r) of year-0-undiscounted annuities, any term.
    assert schedule.pv_per_gbp_of_price(0.03, 0) == pytest.approx(1.03)
    assert fixed.pv_per_gbp_of_price(0.03, 0) == pytest.approx(1.03)
    combined = AnnuitySchedule.combine([(schedule, 100.0), (fixed, 100.0)])
    assert combined.live(15) == pytest.approx(0.5 * 0.7 + 0.5)


def test_each_interconnector_half_runs_to_its_regime_or_life_end():
    settings = interconnector_acquisition_settings(None)
    cables = {"Nemo Link": {"regime": "cap_and_floor", "operational": 2019},
              "MaresConnect": {"regime": "cap_and_floor", "operational": 2030},
              "Britned": {"regime": "exempt_merchant", "operational": None},
              "IFA": {"regime": "use_of_revenues_share", "operational": 1986},
              "Moyle": {"regime": "other_jurisdiction", "operational": None}}
    years = interconnector_remaining_years(cables, settings, 2030)
    assert years["Nemo Link"] == 14.0
    assert years["MaresConnect"] == 25.0
    assert years["Britned"] == 2011 + 40 - 2030
    assert years["IFA"] == IC_ACQUISITION_DEFAULTS["remaining_life_years_by_cable"]["IFA"]
    assert np.isnan(years["Moyle"])
    with pytest.raises(ValueError, match="retired"):
        interconnector_acquisition_settings({"remaining_life_years": 25.0})


def test_a_carried_purchase_pays_each_line_on_its_schedule():
    terms = {"term_years": 17.5, "annuity_schedules": {"legacy": [[10.0, 0.25], [30.0, 0.75]],
                                                       "nationalised_interconnector_gb_half_acquisition": [[14.0, 1.0]],
                                                       "nationalised_acquisition_transaction_costs": [[30.0, 1.0]]}}
    shares = line_live_shares(terms, 2050, 2030)
    assert shares["nationalised_legacy_acquisition_renewables_nuclear"] == pytest.approx(0.75)
    assert shares["nationalised_interconnector_gb_half_acquisition"] == 0.0
    assert shares["nationalised_acquisition_transaction_costs"] == 1.0
    fixed = line_live_shares({"term_years": 30.0}, 2050, 2030)
    assert set(fixed.values()) == {1.0}


def test_plant_at_the_end_of_its_life_is_rebuilt_and_the_law_decides_whose():
    from scripts.public_ownership.legislation_year import REBUILT_METHOD, rebuild_at_end_of_life

    rows = pd.DataFrame([
        _row("on05", "wind_onshore", vintage="legacy", commissioning_year=2005.0, finance_technology="Onshore wind"),
        _row("on02", "wind_onshore", vintage="legacy", commissioning_year=2002.0, finance_technology="Onshore wind"),
        _row("on98", "wind_onshore", vintage="legacy", commissioning_year=1998.0, finance_technology="Onshore wind"),
        _row("gas", "CCGT", vintage="legacy", commissioning_year=2004.0, finance_technology="Gas generation - unabated"),
    ])
    lives = {"Onshore wind": 25.0, "Gas generation - unabated": 25.0}
    rebuilt = rebuild_at_end_of_life(rows, lives, policy_start_year=2025, model_year=2030).set_index("asset_id")
    assert rebuilt.at["on05", "vintage_method"] == REBUILT_METHOD and rebuilt.at["on05", "commissioning_year"] == 2030.0
    assert rebuilt.at["on05", "rebuilt_from_year"] == 2005.0
    # Past its life before the policy start: life-extended before any policy, still existing plant.
    assert rebuilt.at["on98", "finance_vintage"] == "legacy"
    # Gas is life-extended in CP30, not rebuilt.
    assert rebuilt.at["gas", "finance_vintage"] == "legacy"
    judged = apply_legislation_year(rebuilt.reset_index(), legislated_year=2028, policy_start_year=2025,
                                    model_year=2030, transfer_year=2030, leads=LEADS).set_index("asset_id")
    # Rebuilt in 2030 (invested 2028): the state's. Rebuilt in 2027 (invested 2025): private, bought by P3.
    assert judged.at["on05", "finance_vintage"] == "new_build"
    assert judged.at["on02", "finance_vintage"] == "legacy"
    assert judged.at["on02", "commissioning_year"] == 2027.0


def test_the_priced_plan_excludes_small_scale_plant_as_the_scope_plan_does():
    """F28 fix (28 Oct 2026): the purchase's valuation preview carries ownership_exclusion, so rooftop solar the
    scope plan excludes is never valued (check_plan_scope refused every acquiring CP30 register before)."""
    from scripts.public_ownership.calculate_financing_scenarios import _valuation_preview
    from scripts.public_ownership.legacy_valuation import GROUP_EXCLUDED, scope_group, settings_from_config

    legacy = pd.DataFrame([
        {"asset_id": "roof", "component_type": "Generator", "carrier": "solar_pv", "ownership_exclusion": SMALL_SCALE_EXCLUSION},
        {"asset_id": "farm", "component_type": "Generator", "carrier": "solar_pv", "ownership_exclusion": None},
        {"asset_id": "store", "component_type": "StorageUnit", "carrier": "Battery", "ownership_exclusion": None},
    ]).assign(support_type="merchant", cfd_round="", capacity_mw=10.0, dispatch_mwh=1.0e4, fixed_om_gbp_per_kw_year=5.0,
              variable_om_gbp_per_mwh=0.0, finance_technology="Solar PV", model_year=2030, capex_power_gbp_per_kw=500.0,
              market_revenue_gbp=1.0e5, support_revenue_gbp=0.0, fuel_carbon_gbp=0.0)
    settings = settings_from_config({})
    preview = _valuation_preview(legacy, {})
    pd.testing.assert_series_equal(scope_group(preview, settings), scope_group(legacy, settings))
    assert scope_group(preview, settings).tolist()[0] == GROUP_EXCLUDED
    assert scope_group(preview, settings).tolist()[1] != GROUP_EXCLUDED
    # A register without the column (2050 or an older one) keeps its preview as before.
    assert "ownership_exclusion" not in _valuation_preview(legacy.drop(columns="ownership_exclusion"), {}).columns


def test_the_micro_share_comes_from_rows_off_a_contract_first():
    """S3 scope audit (14 Nov 2026): micro plant holds no CfD, so a CfD park keeps its whole capacity ownable."""
    register = pd.DataFrame([
        {"asset_id": "park", "carrier": "solar_pv", "data_source": "FES", "capacity_mw": 60.0, "dispatch_mwh": 6.0,
         "finance_eligible": True, "evidence_status": "resolved", "cohort_share": 1.0, "support_type": "CfD"},
        {"asset_id": "merchant", "carrier": "solar_pv", "data_source": "FES", "capacity_mw": 40.0, "dispatch_mwh": 4.0,
         "finance_eligible": True, "evidence_status": "resolved", "cohort_share": 1.0, "support_type": "merchant"},
    ])
    out = exclude_small_scale(register, micro_mw_by_carrier={"solar_pv": 25.0}).set_index("asset_id")
    assert "park::small_scale" not in out.index and out.at["park", "capacity_mw"] == pytest.approx(60.0)
    assert out.at["merchant::small_scale", "capacity_mw"] == pytest.approx(25.0)
    # More micro than the rows off a contract hold: all rows share it, as before.
    out = exclude_small_scale(register, micro_mw_by_carrier={"solar_pv": 50.0}).set_index("asset_id")
    assert out.at["park::small_scale", "capacity_mw"] == pytest.approx(30.0)


def test_an_expired_modelled_contract_keeps_the_legislation_status_of_its_award():
    """S1/S2 audit (14 Nov 2026): 2031 gas CCS on a DPA is committed at the DPA's award (2027) in 2040, while live, and
    in 2050, once it has expired: in construction, not state new build, in both years."""
    from scripts.public_ownership.legislation_year import eligible_share, load_lead_times

    leads = load_lead_times()
    live = {"carrier": "CCS Gas", "support_type": "DPA", "cfd_round": "DPA_2031", "build_year": 2031,
            "contract_round_at_build": "DPA_2031"}
    expired = {**live, "support_type": "merchant", "cfd_round": ""}
    kw = dict(legislated_year=2028, policy_start_year=2025, snapshot_year=2024, vintaged=True)
    a = eligible_share(live, leads, model_year=2040, **kw)
    b = eligible_share(expired, leads, model_year=2050, **kw)
    assert a["share"] == b["share"] == 0.0 and a["committed_year"] == b["committed_year"] == 2027.0


def test_the_purchase_values_the_ccs_plants_on_their_contract_income():
    """Fix 15 Nov 2026: the purchase's valuation preview carries Net Zero Teesside's DPA payments and Drax BECCS's
    carbon-removal CfD, which the legacy valuation counts (8 Nov 2026, user: P3 buys the CCS plants). Without them
    the purchase priced both plants at a fraction of the owners' value while the owners give the income up."""
    from scripts.public_ownership.calculate_financing_scenarios import _valuation_preview
    from scripts.public_ownership.legacy_valuation import settings_from_config, value_legacy_fleet
    from tests.unit.test_public_ownership_legacy_valuation import _stations

    legacy = pd.DataFrame([
        {"asset_id": "Net Zero Teesside Power::pre_legislation", "carrier": "CCS Gas", "support_type": "DPA",
         "dpa_support_gbp": 4.0e8, "carbon_removal_cfd_revenue_gbp": 0.0, "fuel_carbon_gbp": 1.4e8},
        {"asset_id": "Drax BECCS::pre_legislation", "carrier": "CCS Biomass", "support_type": "CfD",
         "dpa_support_gbp": 0.0, "carbon_removal_cfd_revenue_gbp": 2.25e8, "fuel_carbon_gbp": 4.5e8},
    ]).assign(component_type="Generator", cfd_round="", capacity_mw=600.0, dispatch_mwh=4.0e6, commissioning_year=2027.5,
              fixed_om_gbp_per_kw_year=40.0, variable_om_gbp_per_mwh=4.0, finance_technology="Gas CCS", model_year=2030,
              capex_power_gbp_per_kw=2000.0, market_revenue_gbp=2.2e8, support_revenue_gbp=4.0e8, cfd_support_gbp=0.0)
    preview = _valuation_preview(legacy, {})
    assert preview["dpa_support_gbp"].tolist() == [4.0e8, 0.0]
    assert preview["carbon_removal_cfd_revenue_gbp"].tolist() == [0.0, 2.25e8]
    settings = settings_from_config({})
    with_income = value_legacy_fleet(preview, 0.03, settings, _stations())[0]["pv_state_gbp"].to_numpy()
    without = value_legacy_fleet(preview.drop(columns=["dpa_support_gbp", "carbon_removal_cfd_revenue_gbp"]), 0.03,
                                 settings, _stations())[0]["pv_state_gbp"].to_numpy()
    assert (with_income > without).all() and with_income[0] > without[0] + 1.0e9   # NZT: the DPA over its term
    # A register without the columns keeps its preview as before.
    assert "dpa_support_gbp" not in _valuation_preview(legacy.drop(columns=["dpa_support_gbp"]), {}).columns
