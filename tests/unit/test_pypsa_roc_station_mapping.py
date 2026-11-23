from types import SimpleNamespace

import pandas as pd

from scripts.consumer_bill.map_pypsa_generators_to_ofgem_roc import (
    build_mapping,
    complete_station_representation_mapping,
    transfer_repd_site_metadata,
)


def test_legacy_thermal_identity_uses_exact_site_fingerprint_and_crosswalk():
    network = SimpleNamespace(generators=pd.DataFrame({
        "carrier": ["biomass"],
        "bus": ["B1"],
        "p_nom": [25.0],
        "lat": [52.1],
        "lon": [-1.2],
        "data_source": ["REPD"],
        "support_type": ["merchant"],
    }, index=["gen_biomass_10"]))
    sites = pd.DataFrame({
        "category": ["biomass"],
        "capacity_mw": [25.0],
        "lat": [52.1],
        "lon": [-1.2],
        "source_site_name": ["Real Biomass Station"],
        "repd_ref_id": [123],
        "data_source": ["REPD"],
    })

    network, transfer_summary = transfer_repd_site_metadata(network, sites)
    outturn = pd.DataFrame({
        "Accreditation reference": ["R1"],
        "Station name": ["Ofgem Biomass Station"],
        "Technology": ["Fuelled"],
        "ofgem_roc_certificates": [100.0],
        "ofgem_eligible_generation_mwh": [80.0],
        "implied_roc_band": [1.25],
    })
    registry = pd.DataFrame({
        "Accreditation reference": ["R1"],
        "Station DNC in kW": [25000],
    })
    crosswalk = pd.DataFrame({
        "repd_ref_id": [123],
        "mapping_status": ["high_confidence_exact_name"],
        "ro_accreditation_reference": ["R1"],
        "ro_station_name": ["Ofgem Biomass Station"],
        "name_score": [100.0],
        "capacity_relative_difference": [0.0],
        "capacity_agrees": [True],
    })

    mapping, summary = build_mapping(network, outturn, registry, crosswalk)

    assert transfer_summary["repd_site_metadata_transferred"] == 1
    assert network.generators.at["gen_biomass_10", "source_site_name"] == "Real Biomass Station"
    assert mapping["mapping_status"].item() == "high_confidence_repd_crosswalk"
    assert mapping["actual_roc_band"].item() == 1.25
    assert summary["ofgem_rocs_covered"] == 100.0


def test_residual_stations_receive_explicit_aggregate_representation():
    direct = pd.DataFrame({
        "generator": ["Known station"],
        "mapping_status": ["high_confidence_source_reference"],
        "ro_accreditation_reference": ["R1"],
        "ofgem_roc_certificates": [100.0],
        "p_nom_mw": [20.0],
    })
    outturn = pd.DataFrame({
        "Accreditation reference": ["R1", "R2"],
        "Station name": ["Known station", "Small unnamed wind"],
        "Technology": ["Onshore wind", "Onshore wind"],
        "ofgem_roc_certificates": [100.0, 25.0],
        "ofgem_eligible_generation_mwh": [100.0, 25.0],
        "implied_roc_band": [1.0, 1.0],
    })
    registry = pd.DataFrame({
        "Accreditation reference": ["R1", "R2"],
        "Station DNC in kW": [20000, 500],
    })
    assumptions = pd.DataFrame({
        "ofgem_technology": ["Onshore wind"],
        "representation_id": ["aggregate_onshore_wind"],
        "model_carriers": ["wind_onshore;embedded_wind"],
        "representation_confidence": ["assumed"],
        "assumption": ["Represent residual stations in the aggregate wind fleet."],
    })

    result, summary = complete_station_representation_mapping(
        direct, outturn, registry, assumptions
    )

    assumed = result.loc[result["ro_accreditation_reference"].eq("R2")]
    assert assumed["mapping_status"].item() == "assumed_distributed_aggregate"
    assert assumed["representation_id"].item() == "aggregate_onshore_wind"
    assert summary["direct_identity_certificate_coverage"] == 0.8
    assert summary["total_station_representation_coverage"] == 1.0
    assert summary["total_certificate_representation_coverage"] == 1.0
    assert summary["unrepresented_station_count"] == 0


def test_override_for_accredited_station_without_monthly_output_is_retained_but_not_counted():
    network = SimpleNamespace(generators=pd.DataFrame({
        "carrier": ["biomass"],
        "p_nom": [32.0],
        "support_type": ["ROC"],
    }, index=["Blackburn Meadows Biomass Plant"]))
    outturn = pd.DataFrame({
        "Accreditation reference": ["R_OTHER"],
        "Station name": ["Other issuing station"],
        "Technology": ["Fuelled"],
        "ofgem_roc_certificates": [100.0],
        "ofgem_eligible_generation_mwh": [100.0],
        "implied_roc_band": [1.0],
    })
    registry = pd.DataFrame({
        "Accreditation reference": ["R_OTHER", "R_BLACKBURN"],
        "Station DNC in kW": [1000, 32000],
        "Station name": ["Other issuing station", "Blackburn Meadows Renewable Energy Plant"],
    })
    overrides = pd.DataFrame({
        "generator": ["Blackburn Meadows Biomass Plant"],
        "ro_accreditation_reference": ["R_BLACKBURN"],
        "evidence": ["Accredited station identity"],
    })

    mapping, summary = build_mapping(network, outturn, registry, station_overrides=overrides)

    row = mapping.iloc[0]
    assert row["mapping_status"] == "accredited_no_period_rocs"
    assert row["ro_accreditation_reference"] == "R_BLACKBURN"
    assert row["ofgem_roc_certificates"] == 0.0
    assert summary["high_confidence_generator_matches"] == 0
