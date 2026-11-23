import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.consumer_bill.build_historical_report import build_report_tables
from scripts.consumer_bill.component_ledger import (
    LedgerValidationError,
    add_ledger_ids,
    validate_component_ledger,
    validate_run_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "data" / "market" / "ofgem_price_cap" / "2024_annual"


def test_2024_period_tariff_ledger_reconciles_all_regions_and_periods():
    tables, checks = build_report_tables(
        pd.read_csv(BUNDLE / "period_tariffs.csv"),
        pd.read_csv(BUNDLE / "component_ledger.csv"),
        annual_consumption_kwh=3100,
    )
    assert checks["validated"] is True
    assert checks["period_count"] == 4
    assert checks["region_count"] == 14
    assert checks["maximum_absolute_reconciliation_residual_gbp"] < 0.01
    assert set(tables["regional_bills"]["result_type"]) == {
        "published_tariff_replay",
        "historical_model_reconstruction",
        "household_payment_illustration",
    }
    assert "tax.vat" in set(tables["bill_components"]["component_id"])
    component_totals = tables["bill_components"].groupby(["result_type", "region"])["amount_gbp"].sum()
    bill_totals = tables["regional_bills"].set_index(["result_type", "region"])["bill_gbp_inc_vat"]
    pd.testing.assert_series_equal(
        component_totals.sort_index(), bill_totals.sort_index(), check_names=False, atol=1e-9
    )


def test_negative_amount_is_preserved_as_credit():
    source = pd.read_csv(BUNDLE / "component_ledger.csv").iloc[[0]].copy()
    source["component_id"] = "policy.cfd"
    source["amount_gbp"] = -10.0
    source["rate"] = -10.0
    source["sign"] = "credit"
    source = add_ledger_ids(source.drop(columns="ledger_id"))
    checked = validate_component_ledger(source)
    assert checked.iloc[0]["amount_gbp"] == -10.0


def test_run_manifest_rejects_partition_gap(tmp_path):
    artifact = tmp_path / "one.nc"
    artifact.write_bytes(b"model")
    manifest = {
        "schema_version": "1.0", "run_id": "test",
        "period_start": "2024-01-01T00:00:00Z", "period_end": "2024-01-03T00:00:00Z",
        "artifacts": [{
            "path": str(artifact), "period_start": "2024-01-01T00:00:00Z",
            "period_end": "2024-01-02T00:00:00Z",
        }],
    }
    with pytest.raises(LedgerValidationError, match="do not cover"):
        validate_run_manifest(manifest)


def test_historical_report_rejects_partial_year_or_missing_regions():
    tariffs = pd.read_csv(BUNDLE / "period_tariffs.csv")
    ledger = pd.read_csv(BUNDLE / "component_ledger.csv")
    with pytest.raises(ValueError, match="full calendar year"):
        build_report_tables(
            tariffs.loc[tariffs["cap_period"].eq("2024_Q1")],
            ledger.loc[ledger["cap_period"].eq("2024_Q1")],
        )
    region = tariffs["region"].iloc[0]
    with pytest.raises(ValueError, match="14 charge-restriction regions"):
        build_report_tables(
            tariffs.loc[tariffs["region"].eq(region)], ledger.loc[ledger["region"].eq(region)]
        )


def test_household_profile_does_not_change_tariff_comparison_results():
    profile = pd.DataFrame({
        "timestamp": ["2024-01-01T12:00:00Z"], "consumption_kwh": [3100.0],
    })
    tables, _ = build_report_tables(
        pd.read_csv(BUNDLE / "period_tariffs.csv"), pd.read_csv(BUNDLE / "component_ledger.csv"),
        annual_consumption_kwh=3100, consumption_profile=profile,
    )
    means = tables["regional_bills"].groupby("result_type")["bill_gbp_inc_vat"].mean()
    assert means["published_tariff_replay"] == pytest.approx(means["historical_model_reconstruction"])
    assert means["household_payment_illustration"] != pytest.approx(means["published_tariff_replay"])
    labels = tables["regional_bills"].groupby("result_type")["consumption_profile_basis"].first()
    assert labels["published_tariff_replay"] == "calendar-day allocation proxy"
    assert labels["household_payment_illustration"] == "supplied UTC profile allocated to Europe/London tariff days"


def test_historical_report_records_evidence_without_unapproved_substitution():
    signals = pd.DataFrame({"metric": ["model_demand_mwh"], "value": [1.0], "unit": ["MWh"]})
    procurement = pd.DataFrame({
        "bill_component": ["Published direct fuel energy cost"],
        "candidate_allowance_gbp_per_3100kwh_ex_vat": [100.0],
        "published_allowance_gbp_per_3100kwh_ex_vat": [120.0],
        "recovery_status": ["diagnostic"], "model_basis": ["raw spot backtest"],
    })
    tables, checks = build_report_tables(
        pd.read_csv(BUNDLE / "period_tariffs.csv"), pd.read_csv(BUNDLE / "component_ledger.csv"),
        model_signals=signals,
        cfd_reconciliation={"actual_payment_gbp_mapped_contracts": 100.0, "modelled_payment_independent_formula_gbp": 80.0},
        cfd_mapping={"model_mapping_complete": False}, procurement_bridge=procurement,
    )
    assert checks["model_substitutions"] == 0
    assert checks["model_validation_status"].startswith("evidence supplied")
    assert len(tables["model_validation"]) == 1
    assert tables["support_reconciliation"].iloc[0]["status"] == "incomplete mapping; diagnostic only"
    assert "not substituted" in tables["recovery_bridge"].iloc[0]["status"]
