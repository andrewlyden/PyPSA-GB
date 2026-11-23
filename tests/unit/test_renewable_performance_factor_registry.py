import pandas as pd
import pypsa
import pytest

from scripts.generators.integrate_renewable_generators import add_renewable_generators
from scripts.generators.renewable_performance_factors import (
    resource_region_for_site,
    select_performance_factor,
)


def test_registry_prefers_exact_resource_region_and_month():
    registry = pd.DataFrame([
        {"carrier": "wind_offshore", "resource_region": "GB", "calendar_year": 2020,
         "calendar_month": 1, "factor": 0.64},
        {"carrier": "wind_offshore", "resource_region": "north_sea", "calendar_year": 2020,
         "calendar_month": 1, "factor": 0.78},
        {"carrier": "wind_offshore", "resource_region": "GB", "calendar_year": pd.NA,
         "calendar_month": pd.NA, "factor": 0.65},
    ]).astype({"calendar_year": "Int64", "calendar_month": "Int64"})

    factor, source = select_performance_factor(
        registry, "wind_offshore", "north_sea", 2020, 1, 0.5
    )

    assert factor == pytest.approx(0.78)
    assert source == "registry"


def test_registry_uses_gb_row_when_no_regional_evidence_exists():
    registry = pd.DataFrame([
        {"carrier": "wind_onshore", "resource_region": "GB", "calendar_year": 2020,
         "calendar_month": 1, "factor": 0.84},
    ]).astype({"calendar_year": "Int64", "calendar_month": "Int64"})

    factor, source = select_performance_factor(
        registry, "wind_onshore", "scotland", 2020, 1, 0.5
    )

    assert factor == pytest.approx(0.84)
    assert source == "registry"


def test_resource_regions_are_physical_not_support_type():
    assert resource_region_for_site("wind_offshore", {"lat": 54.0, "lon": 1.0}) == "north_sea"
    assert resource_region_for_site("wind_offshore", {"lat": 54.0, "lon": -4.0}) == "irish_sea"
    assert resource_region_for_site("wind_onshore", {"lat": 56.0, "lon": -3.0}) == "scotland"


def test_integration_stores_registry_factor_independently_of_support_type():
    network = pypsa.Network()
    network.set_snapshots(pd.date_range("2020-01-01", periods=1, freq="h"))
    network.add("Bus", "bus", v_nom=400.0)
    registry = pd.DataFrame([
        {"carrier": "wind_offshore", "resource_region": "north_sea", "calendar_year": 2020,
         "calendar_month": 1, "factor": 0.78},
    ]).astype({"calendar_year": "Int64", "calendar_month": "Int64"})
    sites = pd.DataFrame([{
        "site_name": "Test CfD Wind", "technology": "wind_offshore", "bus": "bus",
        "capacity_mw": 10.0, "lat": 54.0, "lon": 1.0, "support_type": "CfD",
    }])

    add_renewable_generators(
        network, sites, performance_factors={"wind_offshore": 0.64},
        performance_factor_registry=registry,
        performance_factor_context={"calendar_year": 2020, "calendar_month": 1},
    )

    generator = network.generators.iloc[0]
    assert generator["resource_region"] == "north_sea"
    assert generator["renewable_performance_factor"] == pytest.approx(0.78)
    assert generator["renewable_performance_factor_source"] == "registry"
