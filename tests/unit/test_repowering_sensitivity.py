"""OW15: the onshore repowering share at 0.5, report-only (scripts/public_ownership/repowering_sensitivity.py)."""
import pandas as pd
import pytest
import yaml

from scripts.public_ownership import repowering_sensitivity as rs
from scripts.public_ownership.finance import capital_recovery_factor
from config.config_loader import load_defaults


def test_settings_live_outside_valuation_and_need_a_central_rule():
    acquisition = {"valuation": {"repowering_share_by_carrier": {"wind_onshore": 0.3}},
                   "repowering_sensitivity": {"enabled": True, "share": 0.5}}
    assert rs.sensitivity_settings(acquisition) == {"enabled": True, "carrier": "wind_onshore", "share": 0.5}
    with pytest.raises(ValueError, match="no central repowering rule"):
        rs.sensitivity_settings({**acquisition, "repowering_sensitivity": {"enabled": True, "carrier": "solar_pv"}})
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        rs.sensitivity_settings({**acquisition, "repowering_sensitivity": {"enabled": True, "share": 1.5}})
    with pytest.raises(ValueError, match="unknown keys"):
        rs.sensitivity_settings({**acquisition, "repowering_sensitivity": {"enabled": True, "shares": 0.5}})
    assert rs.sensitivity_settings({})["enabled"] is False


def test_the_config_keeps_it_off_under_the_strict_end_of_life_rule():
    # F28 (d), 27 Oct 2026: the central valuation has no private life extension, so there is no repowering
    # share to move; the extension rule is the 2050 toggle group's.
    acquisition = load_defaults()["public_ownership"]["acquisition"]
    assert "repowering_sensitivity" not in acquisition["valuation"]  # valuation refuses unknown keys
    s = rs.sensitivity_settings(acquisition)
    assert not s["enabled"] and (s["carrier"], s["share"]) == ("wind_onshore", 0.5)


def test_tariff_effect_is_the_price_change_at_the_tariffs_annuity_and_flat_rate():
    summary = {"acquisition_terms": {"term_years": 30.0}, "allocation": {"rule_rates": {"final_demand": 1e-8}},
               "conversion": {"vat_rate": 0.05}}
    out = rs.tariff_effect(-1e9, summary, 0.03)
    crf = capital_recovery_factor(0.03, 30.0)
    assert out["annual_charge_change_gbp"] == pytest.approx(-1e9 * crf)
    assert out["bill_change_gbp_per_household_ex_vat"] == pytest.approx(-1e9 * crf * 1e-8)
    assert out["bill_change_gbp_per_household_inc_vat"] == pytest.approx(-1e9 * crf * 1e-8 * 1.05)


def test_valued_market_value_keeps_the_valued_group_floored_at_zero():
    v = pd.DataFrame({"asset_id": ["a", "b", "c"], "scope_group": ["valued", "valued", "thermal_storage"],
                      "pv_market_gbp": [5.0, -2.0, 9.0]})
    assert rs.valued_market_value(v).to_dict() == {"a": 5.0, "b": 0.0}
