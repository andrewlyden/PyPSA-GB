import pandas as pd

from scripts.consumer_bill.build_ofgem_roc_residual_cohorts import build_roc_ledgers


def test_ledger_distinguishes_direct_and_assumed_complete_representation():
    outturn = pd.DataFrame({
        "Accreditation reference": ["R1", "R2"],
        "Station name": ["Named", "Residual"],
        "Scheme": ["RO", "RO"],
        "Country": ["England", "England"],
        "Technology": ["Onshore wind", "Onshore wind"],
        "ofgem_roc_certificates": [80.0, 20.0],
        "ofgem_eligible_generation_mwh": [80.0, 20.0],
        "output_months": [1, 1],
        "implied_roc_band": [1.0, 1.0],
    })
    registry = pd.DataFrame({
        "Accreditation reference": ["R1", "R2"],
        "Station DNC in kW": [10000, 500],
        "Commissioning date": ["01/01/2010", "01/01/2011"],
        "Effective date": ["01/01/2010", "01/01/2011"],
        "Station address": ["A", "B"],
    })
    mapping = pd.DataFrame({
        "mapping_status": [
            "high_confidence_source_reference", "assumed_distributed_aggregate"
        ],
        "ro_accreditation_reference": ["R1", "R2"],
        "generator": ["Named", "__roc_aggregate__aggregate_onshore_wind"],
        "representation_id": ["Named", "aggregate_onshore_wind"],
        "representation_confidence": ["station_evidenced", "assumed"],
        "mapping_evidence": ["Direct evidence", "Aggregate assumption"],
    })

    ledger, cohorts, summary = build_roc_ledgers(outturn, registry, mapping)

    by_ref = ledger.set_index("Accreditation reference")
    assert by_ref.at["R1", "model_representation"] == "named_station_evidenced"
    assert by_ref.at["R2", "model_representation"] == "assumed_distributed_aggregate"
    assert len(cohorts) == 1
    assert summary["named_station_coverage_of_official_certificates"] == 0.8
    assert summary["total_station_representation_coverage"] == 1.0
    assert summary["total_certificate_representation_coverage"] == 1.0
    assert summary["unrepresented_station_count"] == 0
