import pandas as pd
import pytest

from scripts.consumer_bill.build_network_cost_to_bill_bridge import build_network_cost_bridge


def test_network_bridge_uses_model_demand_for_volumetric_charges_only():
    register = pd.DataFrame([
        {"component": "Ofgem Annex 3 — TNUoS", "annual_gbp_per_3100kwh_ex_vat": 10.0},
        {"component": "Ofgem Annex 3 — DUoS unit charges", "annual_gbp_per_3100kwh_ex_vat": 20.0},
        {"component": "Ofgem Annex 3 — DUoS fixed charges", "annual_gbp_per_3100kwh_ex_vat": 30.0},
        {"component": "Ofgem Annex 3 — BSUoS", "annual_gbp_per_3100kwh_ex_vat": 4.0},
        {"component": "Imbalance uplift", "annual_gbp_per_3100kwh_ex_vat": 2.0},
        {"component": "GB-average regional electricity losses", "annual_gbp_per_3100kwh_ex_vat": 5.0},
    ])
    network = pd.DataFrame([
        {"cap_period": "a", "component": "TNUoS", "annual_gbp_per_3100kwh_ex_vat": 5.0, "calendar_weight_2020": 0.5},
        {"cap_period": "b", "component": "TNUoS", "annual_gbp_per_3100kwh_ex_vat": 15.0, "calendar_weight_2020": 0.5},
        {"cap_period": "a", "component": "DUoS unit charges", "annual_gbp_per_3100kwh_ex_vat": 10.0, "calendar_weight_2020": 0.5},
        {"cap_period": "b", "component": "DUoS unit charges", "annual_gbp_per_3100kwh_ex_vat": 30.0, "calendar_weight_2020": 0.5},
        {"cap_period": "a", "component": "DUoS fixed charges", "annual_gbp_per_3100kwh_ex_vat": 20.0, "calendar_weight_2020": 0.5},
        {"cap_period": "b", "component": "DUoS fixed charges", "annual_gbp_per_3100kwh_ex_vat": 40.0, "calendar_weight_2020": 0.5},
    ])
    wholesale = pd.DataFrame({"cap_period": ["a", "b"], "modelled_delivery_demand_share": [0.75, 0.25]})
    recovery = pd.DataFrame([
        {"component": "Ofgem Annex 3 — BSUoS", "candidate_gbp_per_3100kwh_ex_vat": 4.0, "evidence_status": "calibrated_same_year_volume_index", "modelled_signal_unit": "gross", "method_note": "volume"},
        {"component": "Imbalance uplift", "candidate_gbp_per_3100kwh_ex_vat": 2.0, "evidence_status": "calibrated_same_year_volume_index", "modelled_signal_unit": "gross", "method_note": "volume"},
        {"component": "GB-average regional electricity losses", "candidate_gbp_per_3100kwh_ex_vat": 5.0, "evidence_status": "assumption_held_baseline", "modelled_signal_unit": None, "method_note": "baseline"},
    ])

    bridge, summary = build_network_cost_bridge(register, network, wholesale, recovery)

    assert bridge.loc[bridge["component"].eq("Ofgem Annex 3 — TNUoS"), "candidate_allowance_gbp_per_3100kwh_ex_vat"].iloc[0] == pytest.approx(7.5)
    assert bridge.loc[bridge["component"].eq("Ofgem Annex 3 — DUoS unit charges"), "candidate_allowance_gbp_per_3100kwh_ex_vat"].iloc[0] == pytest.approx(15.0)
    assert bridge.loc[bridge["component"].eq("Ofgem Annex 3 — DUoS fixed charges"), "candidate_allowance_gbp_per_3100kwh_ex_vat"].iloc[0] == pytest.approx(30.0)
    assert bridge.loc[bridge["component"].eq("Ofgem Annex 3 — BSUoS"), "same_year_backtest_allowed"].iloc[0]
    assert not bridge.loc[bridge["component"].eq("GB-average regional electricity losses"), "same_year_backtest_allowed"].iloc[0]
    assert summary["same_year_backtest_candidate_total_gbp_per_3100kwh_ex_vat"] == pytest.approx(58.5)
