import numpy as np
import pandas as pd
import pytest

from scripts.consumer_bill.compare_historical_components import compare_components


def inputs():
    register = pd.DataFrame({
        "component": ["Published direct fuel energy cost", "Published CfD allowance embedded in direct fuel",
                      "Ofgem Annex 4 — RO", "Ofgem Annex 3 — BSUoS", "Supplier operating costs", "VAT at 5%"],
        "stack_group": ["hedged_wholesale", "hedged_wholesale", "policy", "network", "supplier_operating", "vat"],
        "annual_gbp_per_3100kwh_ex_vat": [100, 20, 30, 10, 40, 10],
    })
    regional = pd.DataFrame({"result_type": ["published_tariff_replay"] * 14,
        "payment_method": ["direct_debit"] * 14, "meter_type": ["single_rate"] * 14,
        "bill_gbp_ex_vat": [200.] * 14, "bill_gbp_inc_vat": [210.] * 14})
    signals = pd.DataFrame({"metric": ["model_demand_mwh", "cfd_payment_proxy_gbp", "roc_obligation_proxy_gbp"],
                            "value": [310., -1000., 2000.], "unit": ["MWh", "GBP", "GBP"]})
    procurement = pd.DataFrame({"bill_component": ["Published direct fuel energy cost"],
        "candidate_allowance_gbp_per_3100kwh_ex_vat": [80.],
        "recovery_status": ["pypsa_spot_historical_backtest_not_same_year_fitted"], "model_basis": ["spot"]})
    return register, regional, signals, procurement


def test_hybrid_includes_support_credits_and_wholesale_once_with_vat():
    register, regional, signals, procurement = inputs()
    rows, totals, checks = compare_components(None, regional, register=register, signals=signals,
        procurement=procurement, assumptions={"support_recovery_method": "modelled_system_demand_proxy"})
    assert rows.set_index("component").loc["Published CfD allowance embedded in direct fuel", "model_gbp_ex_vat"] == -10
    assert rows.model_gbp_ex_vat.sum() == 90  # 80 wholesale -10 CfD +20 RO
    assert totals.iloc[1].ex_vat_gbp == 140  # +10 published BSUoS +40 operating
    assert totals.iloc[1].inc_vat_gbp == pytest.approx(147)
    assert checks["model_candidate_components"] == 3
    assert not checks["forecast_validation_established"]


def test_calibrated_agreement_is_separate_and_missing_inputs_not_zero():
    register, regional, _, _ = inputs()
    recovery = pd.DataFrame({"component": ["Ofgem Annex 3 — BSUoS"],
        "candidate_gbp_per_3100kwh_ex_vat": [10.], "evidence_status": ["calibrated_same_year_volume_index"],
        "method_note": ["Same-year calibration"], "modelled_signal_value": [0.07], "modelled_signal_unit": ["MWh/MWh"]})
    rows, totals, checks = compare_components(None, regional, register=register, recovery=recovery)
    bsuos = rows.set_index("component").loc["Ofgem Annex 3 — BSUoS"]
    assert np.isnan(bsuos.model_gbp_ex_vat)
    assert bsuos.calibrated_candidate_gbp_ex_vat == 10
    assert checks["calibrated_components"] == 1
    assert checks["model_candidate_components"] == 0
    assert totals.iloc[1].inc_vat_gbp == pytest.approx(210)
    assert rows.loc[rows.stack_group.eq("diagnostic"), "ofgem_gbp_ex_vat"].isna().all()


def test_denominator_must_be_explicit_and_positive():
    register, regional, signals, _ = inputs()
    rows, _, _ = compare_components(None, regional, register=register, signals=signals)
    assert rows.model_gbp_ex_vat.isna().all()
    with pytest.raises(ValueError, match="positive denominator"):
        compare_components(None, regional, register=register, signals=signals,
            assumptions={"support_recovery_method": "eligible_supplier_sales", "eligible_supplier_sales_mwh": 0})


def test_reconciliation_is_visible_and_register_consumption_cannot_be_rescaled():
    register, regional, _, _ = inputs()
    regional["bill_gbp_ex_vat"] = 203.
    regional["bill_gbp_inc_vat"] = 213.15
    rows, totals, _ = compare_components(None, regional, register=register)
    assert rows.set_index("component").loc["Benchmark-to-tariff scope reconciliation", "ofgem_gbp_ex_vat"] == 3.
    assert totals.iloc[1].inc_vat_gbp == pytest.approx(213.15)
    with pytest.raises(ValueError, match="3100 kWh"):
        compare_components(None, regional, register=register, annual_consumption_kwh=2700)


def test_duplicate_bridge_rejected_and_source_data_unchanged():
    register, regional, signals, procurement = inputs()
    original = register.copy(deep=True)
    with pytest.raises(ValueError, match="duplicate"):
        compare_components(None, regional, register=register, procurement=pd.concat([procurement, procurement]))
    pd.testing.assert_frame_equal(register, original)


def test_raw_support_candidate_uses_raw_signal_even_with_reconciled_evidence():
    register, regional, signals, _ = inputs()
    recovery = pd.DataFrame({"component": ["Ofgem Annex 4 — RO"],
        "candidate_gbp_per_3100kwh_ex_vat": [None], "evidence_status": ["reconciled_signal"],
        "method_note": ["Contains historical top-ups"], "modelled_signal_value": [9000.],
        "modelled_signal_unit": ["GBP"]})
    rows, _, _ = compare_components(None, regional, register=register, signals=signals,
        recovery=recovery, assumptions={"support_recovery_method": "modelled_system_demand_proxy"})
    ro = rows.set_index("component").loc["Ofgem Annex 4 — RO"]
    assert ro.model_signal_value == 2000.
    assert ro.model_gbp_ex_vat == 20.
