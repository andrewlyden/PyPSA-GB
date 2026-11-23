import pandas as pd
import pytest

from scripts.consumer_bill.reconcile_lccc_cfd_model_dispatch import (
    _settlement_window_mask,
    _validate_distinct_snapshots,
    calculate_contract_formula_payments,
)


def test_daily_contract_terms_are_applied_to_allocated_physical_dispatch():
    dispatch = pd.DataFrame(
        {"generator": [10.0, 20.0]},
        index=pd.DatetimeIndex(["2020-01-01 12:00", "2020-01-02 12:00"]),
    )
    mapping = pd.DataFrame({
        "CfD_ID": ["contract"],
        "model_generator": ["generator"],
        "allocation_share": [0.5],
    })
    terms = pd.DataFrame({
        "CfD_ID": ["contract", "contract"],
        "settlement_date": pd.DatetimeIndex(["2020-01-01", "2020-01-02"]),
        "formula_payment_gbp_per_mwh": [50.0, -10.0],
    })

    result = calculate_contract_formula_payments(dispatch, mapping, terms).iloc[0]

    # 10 MWh x 50% x GBP50, plus 20 MWh x 50% x -GBP10.
    assert result["modelled_payment_independent_formula_gbp"] == pytest.approx(150.0)
    assert result["modelled_contract_settlement_eligible_dispatch_mwh"] == pytest.approx(15.0)
    assert result["modelled_dispatch_without_price_terms_mwh"] == pytest.approx(0.0)


def test_missing_daily_terms_are_reported_not_silently_filled():
    dispatch = pd.DataFrame(
        {"generator": [10.0]}, index=pd.DatetimeIndex(["2020-01-02 12:00"])
    )
    mapping = pd.DataFrame({
        "CfD_ID": ["contract"],
        "model_generator": ["generator"],
        "allocation_share": [0.5],
    })
    terms = pd.DataFrame({
        "CfD_ID": ["contract"],
        "settlement_date": pd.DatetimeIndex(["2020-01-01"]),
        "formula_payment_gbp_per_mwh": [50.0],
    })

    result = calculate_contract_formula_payments(dispatch, mapping, terms).iloc[0]

    assert result["modelled_dispatch_without_price_terms_mwh"] == pytest.approx(5.0)
    assert result["modelled_contract_settlement_eligible_dispatch_mwh"] == pytest.approx(0.0)


def test_overlapping_cfd_network_timestamps_are_rejected():
    first = pd.date_range("2024-01-01", periods=2, freq="h")
    with pytest.raises(ValueError, match="overlapping snapshots"):
        _validate_distinct_snapshots(first, first, "duplicate.nc")


def test_settlement_window_restricts_model_energy_to_the_settled_days():
    """A part-year contract must not be measured against the whole model period.

    LCCC reports settled generation only for days a contract was live, so
    comparing it against full-period modelled energy understates every derived
    factor. Regression for Moray East / Hornsea 2 in the 2024 backtest.
    """
    index = pd.date_range("2024-01-01", periods=10, freq="D")

    full = _settlement_window_mask(index, None, None)
    assert full.all()

    partial = _settlement_window_mask(index, "2024-01-04", "2024-01-06")
    assert list(index[partial]) == list(pd.DatetimeIndex(
        ["2024-01-04", "2024-01-05", "2024-01-06"]
    ))

    # The last settlement date is inclusive of the whole calendar day.
    hourly = pd.date_range("2024-01-04 00:00", periods=72, freq="h")
    assert _settlement_window_mask(hourly, "2024-01-04", "2024-01-06").sum() == 72

    open_ended = _settlement_window_mask(index, "2024-01-08", None)
    assert index[open_ended].min() == pd.Timestamp("2024-01-08")
    assert index[open_ended].max() == index.max()
