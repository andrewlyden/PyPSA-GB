import pandas as pd

from scripts.consumer_bill.audit_pypsa_roc_capacity_coverage import audit_capacity


class _Network:
    generators = pd.DataFrame({
        "carrier": ["solar_pv", "embedded_solar", "biomass"],
        "p_nom": [10.0, 5.0, 20.0],
        "support_type": ["ROC", None, None],
    }, index=["station__agg2", "embedded_solar", "biomass_station"])


def test_embedded_capacity_is_not_counted_as_station_level_coverage():
    mapping = pd.DataFrame({
        "repd_technology": ["Solar Photovoltaics", "Biomass (dedicated)"],
        "repd_capacity_mw": [10.0, 20.0],
        "mapping_status": ["high_confidence_exact", "review_required"],
    })
    coverage, _, summary = audit_capacity(_Network(), mapping)

    solar = coverage.loc[coverage["carrier"].eq("solar_pv")].iloc[0]
    assert solar["pypsa_explicit_station_capacity_mw"] == 10.0
    assert summary["pypsa_embedded_or_distributed_capacity_mw"] == 5.0
