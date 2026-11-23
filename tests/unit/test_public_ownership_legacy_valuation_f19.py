"""F19 P3 decisions 2 and 3 (user, 6 Oct 2026): the legacy valuation's price path and asset data.

* After the model year, prices follow the declared path (CP30's level), and a CfD's top-up
  falls as its reference rises.
* CfDs run from their start for their round's term, a phase covering its contracted MW.
* Hinkley Point C steps to 3,260 MW from 2031; Sizewell B runs to 2055 with a CfD from 2035;
  Drax's contract years are valued at its own margin; energy from waste earns gate fees;
  an owner closes plant whose cash turns negative.
* Thermal and storage are bought at their owners' value centrally (book value a sensitivity).
* A vintaged (2050) valuation keeps the old rules.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.public_ownership import legacy_valuation as lv
from scripts.public_ownership.levers import LEVER_LEGACY_PRICE_PATH, LEVER_THERMAL_BASIS, lever_default

from tests.unit.test_public_ownership_legacy_valuation import PUBLIC_RATE, _row, _stations
from config.config_loader import load_defaults

ROOT = Path(__file__).resolve().parents[2]


def _settings(**overrides):
    return lv.settings_from_config({"horizon_year": 2070, **overrides})


def _value(rows, **overrides):
    return lv.value_legacy_fleet(pd.DataFrame(rows), PUBLIC_RATE, _settings(**overrides), _stations())


def _profile(rows, **overrides):
    _, profile, _ = _value(rows, **overrides)
    return profile.groupby(["carrier", "stream", "year"])["gbp"].sum()


class TestPricePath:
    def test_years_after_the_model_year_take_the_path_and_top_ups_fall(self):
        row = {**_row("farm", "wind_offshore", "CfD", market=100.0, cfd=300.0, cfd_round="AR2", award=2017,
                      commissioning=2025), "market_revenue_path_gbp": 160.0, "cfd_support_path_gbp": 240.0}
        p = _profile([row])
        assert p[("wind_offshore", "net_operating_cash", 2030)] == 100.0  # the model year is as solved
        assert p[("wind_offshore", "net_operating_cash", 2031)] == 160.0
        assert p[("wind_offshore", "cfd_top_up", 2030)] == 300.0
        assert p[("wind_offshore", "cfd_top_up", 2031)] == 240.0
        # After the contract the merchant tail earns the path's price, not the model year's.
        assert p[("wind_offshore", "net_operating_cash", 2045)] == 160.0
        valuation, _, summary = _value([row])
        assert summary["price_path"] == lv.PRICE_PATH_CP30 and valuation["price_path"].iloc[0] == lv.PRICE_PATH_CP30

    def test_without_path_columns_the_model_price_is_held(self):
        _, _, summary = _value([_row("farm", "wind_offshore", "merchant", market=100.0, commissioning=2025)])
        assert summary["price_path"] == lv.PRICE_PATH_MODEL

    def test_attach_price_path_splits_a_cohort_row_as_the_settlement_does(self):
        rows = pd.DataFrame([{**_row("x::legacy", "wind_onshore", "merchant", market=40.0, cfd=0.0),
                              "cohort_legacy_share": 0.25},
                             {**_row("y", "solar_pv", "merchant", market=7.0), "cohort_legacy_share": np.nan}])
        level = pd.DataFrame({"component_type": ["Generator"], "asset_id": ["x"], "market_revenue_gbp": [400.0],
                              "support_revenue_gbp": [0.0], "fuel_carbon_gbp": [0.0], "cfd_support_gbp": [0.0]})
        out = lv.attach_price_path(rows, level).set_index("asset_id")
        assert out.loc["x::legacy", "market_revenue_path_gbp"] == 100.0
        assert out.loc["y", "market_revenue_path_gbp"] == 7.0  # no record: its own value


class TestContracts:
    def test_lccc_phases_date_the_contract_and_cover_their_share(self):
        starts = pd.DataFrame({"asset_match": ["Big Farm", "Big_Farm"], "contract_start": ["2024-01-01", "2027-07-02"],
                               "contract_capacity_mw": [20.0, 20.0]})
        row = _row("Big_Farm", "wind_offshore", "CfD", market=0.0, cfd=100.0, cfd_round="AR2", award=2017,
                   commissioning=2022)
        valuation, profile, _ = _value([row], cfd_contract_starts=starts)
        v = valuation.iloc[0]
        assert v["cfd_contract_start_basis"] == "lccc_contract_phases"
        assert v["cfd_contracted_share"] == pytest.approx(0.4)  # 40 of 100 MW
        p = profile.set_index(["stream", "year"])["gbp"]
        assert p[("cfd_top_up", 2030)] == pytest.approx(40.0)
        # The first phase ends 1 Jan 2039 (15 years), the second on 2 Jul 2042.
        assert p[("cfd_top_up", 2039)] == pytest.approx(20.0)
        assert p[("cfd_top_up", 2042)] == pytest.approx(20.0 * 182 / 365, rel=1e-3)
        assert ("cfd_top_up", 2043) not in p.index

    def test_round_terms_end_overrides_and_the_award_fallback(self):
        rows = [
            _row("ar7", "wind_offshore", "CfD", cfd=10.0, cfd_round="AR7", award=2025, commissioning=2029),
            _row("undated", "wind_onshore", "CfD", cfd=10.0, cfd_round="AR5", award=2023),
            _row("Drax Power Station (biomass units)", "biomass", "CfD", market=0.0, cfd=10.0, cfd_round="DRAX_BRIDGE",
                 award=2025, commissioning=2013),
        ]
        v = _value(rows, declared_margins=None)[0].set_index("asset_id")
        assert v.loc["ar7", "cfd_contract_end_year"] == pytest.approx(2029.5 + 20)  # 20 years from AR7
        assert v.loc["undated", "cfd_contract_end_year"] == pytest.approx(2023 + 3 + 15)
        assert v.loc["undated", "cfd_contract_start_basis"] == "award_plus_years"
        assert v.loc["Drax Power Station (biomass units)", "cfd_contract_end_year"] == pytest.approx(2031.25)

    def test_the_repository_starts_table_parses_and_names_the_offshore_phases(self):
        table = lv.read_contract_starts(ROOT / lv.DEFAULT_SETTINGS["cfd_contract_starts"])
        seagreen = table.loc[table["key"].eq(lv.normalised_name("Seagreen"))]
        assert seagreen["contract_capacity_mw"].sum() == pytest.approx(454.0)
        hornsea2 = table.loc[table["key"].eq(lv.normalised_name("Hornsea_2___Optimus_and_Breesea"))]
        assert hornsea2["start_year"].min() == pytest.approx(lv.decimal_year("2024-03-28"))


class TestNamedPlant:
    def test_hinkley_point_c_steps_to_both_units_from_2031(self):
        row = {**_row("Hinkley Point C::legacy", "nuclear", "CfD", market=100.0, cfd=200.0, cfd_round="HPC",
                      award=2016), "capacity_mw": 2317.0}
        valuation, profile, _ = _value([row])
        p = profile.set_index(["stream", "year"])["gbp"]
        assert p[("cfd_top_up", 2030)] == 200.0
        assert p[("cfd_top_up", 2031)] == pytest.approx(200.0 * 3260.0 / 2317.0)
        assert valuation["capacity_step_ratio"].iloc[0] == pytest.approx(3260.0 / 2317.0)

    def test_sizewell_b_closes_with_the_fes_fleet_and_its_cfd_starts_in_2035(self):
        # D3 (user, 14 Nov 2026): to FES's 2036 closure centrally (the extended-lives lever keeps EDF's 2055).
        row = {**_row("Sizewell B::legacy", "nuclear", "merchant", market=500.0, fixed_om=100.0),
               "wholesale_mwh": 10.0, "wholesale_revenue_gbp": 450.0}
        valuation, profile, _ = _value([row])
        assert valuation["retirement_year"].iloc[0] == 2036
        p = profile.set_index(["stream", "year"])["gbp"]
        strike = 70.50 * 132.9 / 138.0
        assert ("cfd_top_up", 2034) not in p.index
        assert p[("cfd_top_up", 2035)] == pytest.approx(strike * 10.0 - 450.0)
        assert ("net_operating_cash", 2036) not in p.index

    def test_drax_is_valued_at_its_guidance_midpoint_and_closes_at_its_cfd_end(self):
        # F19 valuation decision (user, 14 Oct 2026): Drax's 2027-31 guidance (GBP 100-200 m a year
        # from its four 645 MW units), the midpoint; it closes in March 2031, with no merchant months.
        row = {**_row("Drax Power Station (biomass units)", "biomass", "CfD", market=100.0, fuel=900.0, cfd=400.0,
                      cfd_round="DRAX_BRIDGE", award=2025, commissioning=2013), "capacity_mw": 1995.0}
        valuation, profile, _ = _value([row])
        margin = 150.0e6 * 1995.0 / 2580.0 * 132.9 / 138.0
        p = profile.set_index(["stream", "year"])["gbp"]
        assert p[("net_operating_cash", 2030)] == pytest.approx(margin)
        assert ("cfd_top_up", 2030) not in p.index  # inside the margin
        # 2031: a quarter at the margin and nothing after it (closed, not a merchant losing 800).
        assert p[("net_operating_cash", 2031)] == pytest.approx(0.25 * margin)
        assert ("net_operating_cash", 2032) not in p.index
        v = valuation.iloc[0]
        assert v["closure_year"] == pytest.approx(2031.25) and v["retirement_year"] == pytest.approx(2031.25)
        assert v["retirement_method"] == "closed_at_declared_margin_end"
        assert v["retirement_year_before_closure"] == 2038  # biomass life 25 from 2013, kept for book value

    def test_a_declared_margin_without_close_at_end_keeps_the_merchant_tail(self):
        margins = {"Drax Power Station (biomass units)": {
            "segment_ebitda_gbp": 150.0e6, "segment_capacity_mw": 2580.0, "through_year": 2031.25,
            "rebasing_to_model_price_year": 1.0, "close_at_end": False}}
        row = {**_row("Drax Power Station (biomass units)", "biomass", "CfD", market=100.0, fuel=900.0, cfd=400.0,
                      cfd_round="DRAX_BRIDGE", award=2025, commissioning=2013), "capacity_mw": 1995.0}
        valuation, profile, _ = _value([row], declared_margins=margins)
        margin = 150.0e6 * 1995.0 / 2580.0
        p = profile.set_index(["stream", "year"])["gbp"]
        assert p[("net_operating_cash", 2031)] == pytest.approx(0.25 * margin - 0.75 * 800.0)
        assert valuation["closure_year"].iloc[0] == 2032

    def test_energy_from_waste_earns_its_gate_fee(self):
        row = {**_row("efw", "waste_to_energy", "merchant", market=10.0), "dispatch_mwh": 1000.0}
        valuation, _, _ = _value([row])
        data = json.loads((ROOT / "data/public_ownership/financing/efw_gate_fees_2050.json").read_text())
        per_mwh = data["gate_fee_gbp_per_tonne"]["central"] * 3.6 / (9.62 * 0.25)
        assert valuation["gate_fee_model_year_gbp"].iloc[0] == pytest.approx(1000.0 * per_mwh)
        assert valuation["net_operating_cash_model_year_gbp"].iloc[0] == pytest.approx(10.0 + 1000.0 * per_mwh)
        # A register that already books gate fees (2050's support line) is never credited twice.
        booked = {**row, "efw_gate_fee_revenue_gbp": 5.0}
        assert _value([booked])[0]["gate_fee_model_year_gbp"].iloc[0] == 0.0

    def test_a_closed_plant_keeps_its_life_for_book_value(self):
        row = _row("ccgt::legacy", "CCGT", "merchant", market=10.0, fuel=50.0, commissioning=2010)
        v = _value([row])[0].iloc[0]
        assert v["retirement_method"] == "closed_when_cash_negative" and v["retirement_year"] == 2031
        assert v["retirement_year_before_closure"] == 2035   # 2010 + 25 (gas life, continuity audit 14 Nov 2026)


def test_a_vintaged_valuation_keeps_the_old_rules():
    rows = [{**_row("Hinkley Point C::legacy", "nuclear", "CfD", market=100.0, cfd=200.0, cfd_round="AR1",
                    award=2015, commissioning=2040), "model_year": 2050},
            {**_row("ccgt::legacy", "CCGT", "merchant", market=10.0, fuel=50.0, commissioning=2040), "model_year": 2050}]
    v = _value(rows)[0].set_index("asset_id")
    assert v.loc["Hinkley Point C::legacy", "cfd_contract_end_year"] == 2015 + 18
    assert "closure_year" not in v.columns and "gate_fee_model_year_gbp" not in v.columns
    assert v.loc["ccgt::legacy", "retirement_method"] != "closed_when_cash_negative"


def test_the_config_valuation_block_equals_the_code_defaults():
    block = load_defaults()
    valuation = block["public_ownership"]["acquisition"]["valuation"]
    settings = lv.settings_from_config(valuation)
    for key in ("cfd_contract_end_year_overrides", "cfd_contract_starts", "cfd_contract_term_years_by_round",
                "cfd_contract_term_years_default", "cfd_start_years_after_award_undated", "capacity_from_year",
                "new_contracts", "declared_margins", "gate_fee_income", "close_when_cash_negative",
                "retirement_year_overrides"):
        assert valuation[key] == lv.DEFAULT_SETTINGS[key], key
    # D3 (user, 14 Nov 2026): FES's 2036 closure centrally; EDF's 2055 under the extended-lives lever.
    assert settings["retirement_year_overrides"]["Sizewell B"] == 2036
    assert block["public_ownership"]["acquisition"]["extended_asset_lives"]["retirement_year_overrides"] == {
        "Sizewell B": 2055}


def test_the_lever_defaults_are_owners_value_and_the_model_price_path():
    # F29 (user, 26 Oct 2026): the model's own 2030 price is central, CP.32 the sensitivity.
    assert lever_default(LEVER_THERMAL_BASIS) == "market_value"
    # User decision (10 Nov 2026): the anchored path (today, 2030, 2050) is central; 2030 held flat a named case.
    assert lever_default(LEVER_LEGACY_PRICE_PATH) == lv.PRICE_PATH_ANCHORED


def test_the_purchase_basis_is_labelled_group_by_group():
    from scripts.public_ownership.calculate_financing_scenarios import purchase_basis_label

    assert purchase_basis_label("market_value", "market_value") == (
        "renewables and nuclear at owners' value (after tax, workbook A2), thermal and storage at owners' value")
    assert purchase_basis_label("market_value", "zero").endswith("thermal and storage at no price")
    assert purchase_basis_label("book_value", "market_value") == "the whole fleet at net book value (workbook A1)"


def test_transaction_check_reads_the_owners_value_per_mw():
    valuation = pd.DataFrame({"scope_group": ["valued"], "carrier": ["wind_offshore"], "capacity_mw": [100.0],
                              "pv_market_gbp": [200.0e6], "pv_state_gbp": [250.0e6]})
    multiples = pd.read_csv(ROOT / "data/public_ownership/financing/legacy_transaction_multiples.csv")
    rows = lv.transaction_check(valuation, multiples)
    offshore = [r for r in rows if r["carrier"] == "wind_offshore"]
    assert offshore and offshore[0]["model_owners_value_gbp_m_per_mw"] == pytest.approx(2.0)
    assert offshore[0]["model_to_transaction"] == pytest.approx(2.0 / offshore[0]["transaction_gbp_m_per_mw"])


def test_a_gas_station_runs_at_least_to_its_capacity_market_agreement_end(tmp_path):
    """Continuity audit (14 Nov 2026): 25 years (GHD for DESNZ 2025), kept to a later Capacity Market agreement's end."""
    schedule = tmp_path / "gas.csv"
    pd.DataFrame({"station_name": ["Old CCGT", "Short CCGT"],
                  "cm_agreements": ["X1 500.0 MW (2028-29 T-4, 15 y)", "X2 300.0 MW (2029-30 T-4, 1 y)"]}
                 ).to_csv(schedule, index=False)
    assert lv.capacity_market_agreement_ends(schedule) == {"oldccgt": 2043.0, "shortccgt": 2030.0}
    rows = pd.DataFrame([_row("Old CCGT", "CCGT", "merchant", market=100.0, commissioning=1997),
                         _row("Short CCGT", "CCGT", "merchant", market=100.0, commissioning=2010)])
    settings = lv.settings_from_config({"cm_agreement_floor": {"file": str(schedule), "carriers": ["CCGT"]},
                                        "horizon_year": 2070})
    years, method = lv.retirement_years(rows, settings)
    assert list(years) == [2043.0, 2035.0]
    assert list(method) == ["capacity_market_agreement_end", "commissioning_plus_life"]


def test_undated_batteries_take_the_fleet_build_profile(tmp_path):
    """Continuity audit (14 Nov 2026): rows of a profiled carrier get build years whose MW reproduce the profile."""
    profile = tmp_path / "p.csv"
    pd.DataFrame({"build_year": [2020, 2026], "mw": [250.0, 750.0]}).to_csv(profile, index=False)
    rows = pd.DataFrame({"asset_id": [f"b{i}" for i in range(4)], "carrier": ["Battery"] * 4,
                         "capacity_mw": [250.0] * 4})
    out = lv.profiled_commissioning(rows, pd.Series([2026.5] * 4), {"Battery": profile})
    assert list(out) == [2020.0, 2026.0, 2026.0, 2026.0]
