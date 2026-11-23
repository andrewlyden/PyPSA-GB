"""Item N0: the legacy fleet's cash flows are valued as a declining profile, not a level annuity.

Small synthetic register and Ofgem-like station list so every expiry rule is
visible: the RO run-off from effective dates and the 2037 closure, CfD expiry at
the contract start plus its round's term (F19, 6 Oct 2026; award plus eighteen
before), retirement at commissioning plus life, named overrides for the nuclear
stations, an owner closing loss-making plant, and the scope groups that keep
thermal and foreign plant out of the headline.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.public_ownership.finance import capital_recovery_factor
from scripts.public_ownership.legacy_valuation import (
    DEFAULT_SETTINGS, GROUP_EXCLUDED, GROUP_THERMAL, GROUP_VALUED, RO_SHARE_ALL,
    annuitised_purchase_cost, present_value, ro_accreditation_shares, settings_from_config,
    value_legacy_fleet,
)
from config.config_loader import load_defaults

MODEL_YEAR = 2030
PUBLIC_RATE = 0.02745


def _settings(**overrides):
    settings = settings_from_config({"horizon_year": 2070, **overrides})
    return settings


def _row(asset_id, carrier, support, *, market=0.0, fuel=0.0, fixed_om=0.0, cfd=0.0, roc=0.0,
         comp=0.0, cfd_round=None, award=None, commissioning=None, component="Generator", wacc=0.07,
         vintage="legacy", case="private"):
    return {
        "case_id": case, "asset_id": asset_id, "component_type": component, "carrier": carrier,
        "finance_vintage": vintage, "support_type": support, "cfd_round": cfd_round, "cfd_award_year": award,
        "capacity_mw": 100.0, "commissioning_year": commissioning, "market_revenue_gbp": market,
        "cfd_support_gbp": cfd, "cfd_curtailment_compensation_gbp": comp, "roc_support_gbp": roc,
        "fuel_carbon_gbp": fuel, "annual_fixed_om_gbp": fixed_om, "wacc": wacc, "model_year": MODEL_YEAR,
    }


def _register():
    return pd.DataFrame([
        # ROC onshore wind commissioned 2012: alive to 2037, ROC income runs off with the register.
        _row("onshore_roc::legacy", "wind_onshore", "ROC", market=100.0, fixed_om=20.0, roc=60.0, commissioning=2012),
        # Investment Contract offshore wind commissioned 2017: a 15-year contract from mid-2017 ends
        # mid-2032 (half of 2032 paid); plant to 2047 (30-year life, decision 48).
        _row("offshore_ic", "wind_offshore", "CfD", market=200.0, fixed_om=50.0, cfd=300.0, comp=40.0,
             cfd_round="Investment Contract", award=2014, commissioning=2017),
        # Hinkley Point C: flat to 2065 by the named override, contract to 2065 by the round override.
        _row("Hinkley Point C::legacy", "nuclear", "CfD", market=800.0, fuel=100.0, fixed_om=300.0, cfd=900.0,
             cfd_round="HPC", award=2016),
        # Loss-making legacy gas: thermal group, negative flow, so its owner closes it after the model year.
        _row("ccgt::legacy", "CCGT", "merchant", market=100.0, fuel=150.0, fixed_om=30.0),
        # Foreign plant: excluded.
        _row("EU_market_FR", "EU_market", "merchant", market=500.0),
        # Not legacy, not private: must be ignored.
        _row("new_offshore", "wind_offshore", "CfD", market=1.0, vintage="new_build"),
        _row("onshore_roc::legacy", "wind_onshore", "ROC", market=1.0, roc=1.0, commissioning=2012, case="public_ownership"),
    ])


def _stations():
    """Two onshore stations accredited 2012 and 2020, one PV station 2015, all 1 MW."""
    return pd.DataFrame({
        "Technology": ["Onshore wind", "Onshore wind", "Photovoltaic"],
        "Station DNC in kW": ["1,000", "1,000", "1,000"],
        "Effective date": ["01/04/2012", "01/04/2020", "01/10/2015"],
    })


class TestRunOff:
    def test_ro_shares_follow_effective_dates_and_the_2037_closure(self):
        shares = ro_accreditation_shares(_stations(), MODEL_YEAR, 2040)
        onshore = shares["wind_onshore"]
        # Both onshore stations live on 1 April 2030; the 2012 one ends 1 April 2032 (20 y), so 2032 shows 0.5.
        assert onshore[2030] == 1.0 and onshore[2031] == 1.0 and onshore[2032] == 0.5
        # The 2020 station would run to 2040 but the scheme closes 31 March 2037.
        assert onshore[2036] == 0.5 and onshore[2037] == 0.0
        assert shares["solar_pv"][2035] == 1.0 and shares["solar_pv"][2036] == 0.0
        assert shares[RO_SHARE_ALL][2030] == 1.0 and shares.attrs["accredited_dnc_mw_model_year"] == pytest.approx(3.0)

    def test_present_value_matches_a_closed_form_annuity(self):
        flows = np.full((1, 31), 10.0)
        pv = present_value(flows, 0.05)[0]
        annuity_due = 10.0 * (1 - 1.05 ** -31) / 0.05 * 1.05
        assert pv == pytest.approx(annuity_due)
        per_row = present_value(np.vstack([flows, flows]), np.array([0.05, 0.10]))
        assert per_row[0] == pytest.approx(pv) and per_row[1] < per_row[0]

    def test_annuitised_purchase_cost_uses_the_capital_recovery_factor(self):
        assert annuitised_purchase_cost(67.1e9, PUBLIC_RATE, 30) == pytest.approx(67.1e9 * capital_recovery_factor(PUBLIC_RATE, 30))


class TestValuation:
    @pytest.fixture(scope="class")
    def result(self):
        # The synthetic Hinkley Point C is 100 MW, so its unit-2 step (to 3,260 MW, F19) is left out
        # here; tests/unit/test_public_ownership_legacy_valuation_f19.py covers it.
        return value_legacy_fleet(_register(), PUBLIC_RATE, _settings(capacity_from_year=None), _stations())

    def test_only_private_legacy_rows_are_valued_and_grouped(self, result):
        valuation, _, summary = result
        assert set(valuation["asset_id"]) == {"onshore_roc::legacy", "offshore_ic", "Hinkley Point C::legacy",
                                              "ccgt::legacy", "EU_market_FR"}
        groups = valuation.set_index("asset_id")["scope_group"]
        assert groups["ccgt::legacy"] == GROUP_THERMAL and groups["EU_market_FR"] == GROUP_EXCLUDED
        assert groups["offshore_ic"] == GROUP_VALUED and groups["Hinkley Point C::legacy"] == GROUP_VALUED
        assert summary["headline_group"] == GROUP_VALUED
        assert summary["by_group"][GROUP_EXCLUDED]["pv_state_gbp"] > 0  # profiled, never in the headline

    def test_expiry_rules(self, result):
        valuation, _, _ = result
        v = valuation.set_index("asset_id")
        assert v.loc["onshore_roc::legacy", "retirement_year"] == 2037 and v.loc["onshore_roc::legacy", "retirement_method"] == "commissioning_plus_life"
        assert v.loc["offshore_ic", "cfd_contract_end_year"] == pytest.approx(2032.5)
        assert v.loc["offshore_ic", "cfd_contract_start_basis"] == "commissioning_mid_year"
        assert v.loc["offshore_ic", "retirement_year"] == 2047
        assert v.loc["Hinkley Point C::legacy", "retirement_year"] == 2065 and v.loc["Hinkley Point C::legacy", "retirement_method"] == "named_override"
        assert v.loc["Hinkley Point C::legacy", "cfd_contract_end_year"] == 2065
        # Undated CCGT with no dated peer would take its carrier's declared remaining years, but it
        # loses money every year, so its owner closes it in the first year after the model year.
        expected = DEFAULT_SETTINGS["undated_remaining_years_by_carrier"].get(
            "CCGT", DEFAULT_SETTINGS["undated_remaining_years_default"])
        assert v.loc["ccgt::legacy", "retirement_year_before_closure"] == MODEL_YEAR + expected
        assert v.loc["ccgt::legacy", "retirement_year"] == MODEL_YEAR + 1
        assert v.loc["ccgt::legacy", "retirement_method"] == "closed_when_cash_negative"

    def test_streams_decline_with_their_own_expiries(self, result):
        _, profile, _ = result
        p = profile.set_index(["carrier", "stream", "year"])["gbp"]
        # Offshore: top-up 300 to 2031, half of it in 2032 (the contract ends mid-year), nothing from
        # 2033; net cash 150 while the contract runs, then 110 to 2046: the 40 of curtailment
        # compensation inside market revenue ends with the CfD.
        assert p[("wind_offshore", "cfd_top_up", 2031)] == 300.0
        assert p[("wind_offshore", "cfd_top_up", 2032)] == pytest.approx(150.0)
        assert ("wind_offshore", "cfd_top_up", 2033) not in p.index
        assert p[("wind_offshore", "net_operating_cash", 2031)] == 150.0
        assert p[("wind_offshore", "net_operating_cash", 2032)] == pytest.approx(130.0)
        assert p[("wind_offshore", "net_operating_cash", 2046)] == 110.0
        assert ("wind_offshore", "net_operating_cash", 2047) not in p.index
        # Onshore ROC income follows the run-off: 60 in 2031, 30 from 2032, 0 in 2037.
        assert p[("wind_onshore", "roc_income", 2031)] == 60.0
        assert p[("wind_onshore", "roc_income", 2032)] == 30.0
        assert ("wind_onshore", "roc_income", 2037) not in p.index
        # Thermal net cash is negative: settled as solved in the model year, closed after it.
        assert p[("CCGT", "net_operating_cash", 2030)] == -80.0
        assert ("CCGT", "net_operating_cash", 2031) not in p.index

    def test_without_the_closure_option_a_loss_runs_to_retirement(self):
        _, profile, _ = value_legacy_fleet(_register(), PUBLIC_RATE, _settings(close_when_cash_negative=False),
                                           _stations())
        p = profile.set_index(["carrier", "stream", "year"])["gbp"]
        last = MODEL_YEAR + DEFAULT_SETTINGS["undated_remaining_years_by_carrier"]["CCGT"]
        assert p[("CCGT", "net_operating_cash", last - 1)] == -80.0
        assert ("CCGT", "net_operating_cash", last) not in p.index

    def test_present_values_and_the_state_owner_gap(self, result):
        valuation, _, summary = result
        v = valuation.set_index("asset_id")
        # HPC: flat 1,300 a year 2030-2064 (35 flows), PV at the public rate as an annuity due.
        pv = 1300.0 * (1 - (1 + PUBLIC_RATE) ** -35) / PUBLIC_RATE * (1 + PUBLIC_RATE)
        assert v.loc["Hinkley Point C::legacy", "pv_state_gbp"] == pytest.approx(pv)
        assert v.loc["Hinkley Point C::legacy", "pv_market_gbp"] < pv  # 7% owner discounts harder
        assert summary["valued_gap_state_minus_market_gbp"] > 0
        assert summary["valued_pv_state_gbp"] == pytest.approx(
            float(valuation.loc[valuation["scope_group"].eq(GROUP_VALUED), "pv_state_gbp"].sum()))
        # Compensation is carried for information and never added as a stream of its own
        # (under contract-value bids it is already inside market revenue).
        assert v.loc["offshore_ic", "cfd_curtailment_compensation_model_year_gbp_not_valued"] == 40.0
        assert v.loc["offshore_ic", "annual_flow_model_year_gbp"] == 450.0
        # The by-stream PVs add up to the total.
        streams = sum(v.loc["offshore_ic", f"pv_state_{s}_gbp"] for s in ("net_operating_cash", "cfd_top_up", "roc_income"))
        assert streams == pytest.approx(v.loc["offshore_ic", "pv_state_gbp"])

    def test_compensation_outside_market_revenue_is_left_alone(self):
        from scripts.public_ownership.legacy_valuation import compensation_in_market_revenue

        frame = pd.DataFrame([
            # Contract-value bids: support is CfD + RO only, so the compensation is in market revenue.
            {**_row("a", "wind_offshore", "CfD", market=200.0, cfd=300.0, comp=40.0), "support_revenue_gbp": 300.0},
            # Otherwise support carries it and market revenue does not.
            {**_row("b", "wind_offshore", "CfD", market=160.0, cfd=300.0, comp=40.0), "support_revenue_gbp": 340.0},
        ])
        assert list(compensation_in_market_revenue(frame)) == [40.0, 0.0]

    def test_summary_is_json_serialisable(self, result):
        json.dumps(result[2])


class TestSettings:
    def test_unknown_keys_and_bad_values_are_refused(self):
        with pytest.raises(ValueError, match="unknown keys"):
            settings_from_config({"purchase_price": 1})
        with pytest.raises(ValueError, match="horizon_year"):
            settings_from_config({"horizon_year": 2029})
        with pytest.raises(ValueError, match="reference_owner_rates"):
            settings_from_config({"reference_owner_rates": [1.5]})

    def test_nested_overrides_merge_over_defaults(self):
        settings = settings_from_config({"retirement_year_overrides": {"Torness": 2030}})
        assert settings["retirement_year_overrides"]["Hinkley Point C"] == 2065
        assert settings["retirement_year_overrides"]["Torness"] == 2030

    def test_repository_defaults_parse(self):
        import yaml
        block = load_defaults()["public_ownership"]["acquisition"]
        settings = settings_from_config(block.get("valuation"))
        assert Path(settings["ro_register"]).exists()

    def test_decision_48_lives_and_the_code_defaults_agree(self):
        """One lifetime rule (user decision 48, 3 Oct 2026): DESNZ lives, onshore 25, pumped hydro past 2050."""
        import yaml
        valuation = load_defaults()["public_ownership"]["acquisition"]["valuation"]
        for key in ("retirement_life_years_by_carrier", "undated_remaining_years_by_carrier"):
            for carrier, value in valuation[key].items():
                assert DEFAULT_SETTINGS[key][carrier] == value, (key, carrier)
        lives = DEFAULT_SETTINGS["retirement_life_years_by_carrier"]
        assert (lives["solar_pv"], lives["wind_offshore"], lives["wind_onshore"], lives["large_hydro"]) == (38, 30, 25, 60)
        assert MODEL_YEAR + DEFAULT_SETTINGS["undated_remaining_years_by_carrier"]["pumped_hydro"] > 2050


@pytest.mark.skipif(not Path("resources/public_ownership/cp30_2030_cp30_network_asset_finance_register.parquet").exists(),
                    reason="needs the completed CP30 2030 finance register")
def test_real_register_reproduces_the_ro_eligible_capacity():
    """The Ofgem register gives 22.85 GW accredited on 1 April 2030, the figure the bill's RO line uses."""
    stations = pd.read_csv(DEFAULT_SETTINGS["ro_register"], low_memory=False)
    shares = ro_accreditation_shares(stations, 2030, 2040)
    assert shares.attrs["accredited_dnc_mw_model_year"] == pytest.approx(22_852.5, rel=1e-3)
    assert shares[RO_SHARE_ALL][2037] == 0.0
