import pandas as pd
import pytest

from scripts.consumer_bill.audit_renewable_performance_factors import (
    build_contract_factor_diagnostics,
    parse_factor_arguments,
    resolve_performance_factors,
)


def test_flat_factor_takes_precedence_over_stacked_components():
    factors, source = resolve_performance_factors({
        "enabled": True,
        "factors": {"wind_offshore": 0.64},
        "era5_bias": {"wind_offshore": 0.8},
    })

    assert source == "flat_override"
    assert factors == {"wind_offshore": 0.64}


def test_repeatable_factor_arguments_build_flat_config():
    config = parse_factor_arguments(["wind_offshore=0.64", "wind_onshore=0.84"])

    assert config == {
        "enabled": True,
        "factors": {"wind_offshore": 0.64, "wind_onshore": 0.84},
    }


def test_contract_factor_diagnostic_recovers_implied_factor():
    reconciliation = pd.DataFrame({
        "CfD_ID": ["contract"],
        "actual_generation_mwh": [90.0],
        "modelled_contract_available_mwh_full_period": [60.0],
    })
    mapping = pd.DataFrame({
        "CfD_ID": ["contract"],
        "mapping_status": ["mapped"],
        "model_carrier": ["wind_offshore"],
    })

    result = build_contract_factor_diagnostics(
        reconciliation, mapping, {"wind_offshore": 0.6}
    ).iloc[0]

    assert result["inferred_pre_factor_available_mwh"] == pytest.approx(100.0)
    assert result["implied_factor_to_match_actual_available_energy"] == pytest.approx(0.9)
    assert result["factor_ratio_to_configured"] == pytest.approx(1.5)
    assert result["interpretation"] == "modelled_available_energy_low_for_contract"
