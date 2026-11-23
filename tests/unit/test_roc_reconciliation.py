import pandas as pd
import pytest

from scripts.consumer_bill.reconcile_roc_model_dispatch import reconcile, reconcile_networks


class _Network:
    snapshots = pd.DatetimeIndex(["2020-01-01 00:00", "2020-01-01 01:00"])
    snapshot_weightings = pd.DataFrame({"generators": [1.0, 1.0]}, index=snapshots)
    generators = pd.DataFrame({
        "carrier": ["wind_onshore", "biomass"],
        "support_type": ["ROC", None],
        "ro_banding": [0.9, None],
        "roc_label_source": ["ofgem_calendar_certificate_outturn", None],
    }, index=["wind", "biomass"])
    generators_t = type("GeneratorTimeSeries", (), {
        "p": pd.DataFrame({"wind": [10.0, 10.0], "biomass": [20.0, 20.0]}, index=snapshots)
    })()


def test_reconciliation_keeps_unlabelled_roc_candidate_dispatch():
    reference = pd.DataFrame({
        "obligation_period": ["2020-21", "2020-21"],
        "scope": ["Great Britain", "Great Britain"],
        "technology": ["onshore_wind", "fuelled"],
        "rocs_issued": [18.0, 30.0],
    })

    result, summary = reconcile(_Network(), reference)

    wind = result.loc[result["technology"].eq("onshore_wind")].iloc[0]
    fuelled = result.loc[result["technology"].eq("fuelled")].iloc[0]
    assert wind["modelled_labelled_roc_certificates"] == 18.0
    assert fuelled["modelled_labelled_roc_certificates"] == 0.0
    assert fuelled["unlabelled_roc_compatible_dispatch_mwh"] == 40.0
    assert fuelled["aggregate_recovery_bridge_roc_certificates"] == 30.0
    assert fuelled["reconciled_roc_certificates"] == 30.0
    assert summary["comparison_status"] == "calendar_aligned_station_labels_with_explicit_recovery_bridge"
    assert summary["direct_model_coverage_of_ofgem_issue"] == 18.0 / 48.0
    assert summary["recovery_bridge_is_calibration"] is True


def test_calendar_reference_splits_model_recovery_from_external_bridge():
    reference = pd.DataFrame({
        "Technology": ["Onshore wind", "Fuelled"],
        "Country": ["England", "England"],
        "ofgem_roc_certificates": [18.0, 30.0],
        "ofgem_eligible_generation_mwh": [20.0, 24.0],
    })

    result, summary = reconcile(_Network(), reference)

    fuelled = result.loc[result["technology"].eq("fuelled")].iloc[0]
    assert fuelled["ofgem_effective_roc_band"] == 1.25
    assert fuelled["calibrated_eligible_share_of_unlabelled_dispatch"] == 0.6
    assert fuelled["recovered_from_unlabelled_model_roc_certificates"] == 30.0
    assert fuelled["external_representation_bridge_roc_certificates"] == 0.0
    assert summary["reconciled_roc_certificates"] == 48.0


def test_distinct_monthly_networks_are_summed_before_calendar_reconciliation():
    reference = pd.DataFrame({
        "technology": ["onshore_wind", "fuelled"],
        "rocs_issued": [36.0, 60.0],
    })

    first, second = _Network(), _Network()
    second.snapshots = second.snapshots + pd.DateOffset(months=1)
    second.snapshot_weightings = second.snapshot_weightings.set_axis(second.snapshots)
    second.generators_t = type("GeneratorTimeSeries", (), {
        "p": second.generators_t.p.set_axis(second.snapshots)
    })()
    result, summary = reconcile_networks([first, second], reference)

    wind = result.loc[result["technology"].eq("onshore_wind")].iloc[0]
    fuelled = result.loc[result["technology"].eq("fuelled")].iloc[0]
    assert wind["modelled_labelled_roc_dispatch_mwh"] == 40.0
    assert wind["modelled_labelled_roc_certificates"] == 36.0
    assert fuelled["unlabelled_roc_compatible_dispatch_mwh"] == 80.0
    assert summary["solved_networks_aggregated"] == 2


def test_overlapping_monthly_networks_are_rejected_before_aggregation():
    reference = pd.DataFrame({"technology": ["onshore_wind"], "rocs_issued": [18.0]})
    with pytest.raises(ValueError, match="overlapping snapshots"):
        reconcile_networks([_Network(), _Network()], reference)
