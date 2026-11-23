import pandas as pd

from scripts.generators import integrate_thermal_generators as thermal
from scripts.renewables.prepare_renewable_site_data import RenewableSiteDataProcessor


def test_historical_renewable_overlay_is_converted_to_repd_schema(monkeypatch):
    source = pd.DataFrame({
        "modelled_year": [2024], "technology": ["Wind Onshore"],
        "repd_ref_id": ["LCCC-C1"], "site_name": ["New wind"],
        "source_site_name": ["New wind"], "capacity_mw": [30.0],
        "x_coord": [100000], "y_coord": [200000],
        "operational_date": ["04/04/2024"], "cfd_round": ["AR1"],
        "cfd_capacity_mw": [29.8], "support_type": ["CfD"], "cfd_id": ["C1"],
        "source_evidence": ["test"], "notes": ["test"],
    })
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: source.copy())
    processor = RenewableSiteDataProcessor()
    monkeypatch.setattr(processor, "convert_coordinates", lambda x, y: (55.0, -3.0))

    result = processor.load_historical_site_overrides("overlay.csv")

    assert result.loc[0, "Site Name"] == "New wind"
    assert result.loc[0, "Technology Type"] == "Wind Onshore"
    assert result.loc[0, "Installed Capacity (MWelec)"] == 30.0
    assert result.loc[0, "Development Status"] == "Operational"


def test_historical_thermal_overlay_filters_to_modelled_year(monkeypatch):
    source = pd.DataFrame({
        "modelled_year": [2023, 2024], "station_name": ["old", "new"],
        "source_site_name": ["old", "new"], "capacity_mw": [1.0, 287.516],
        "fuel_type": ["Biomass (dedicated)", "Biomass (dedicated)"],
        "x_coord": [1, 2], "y_coord": [1, 2],
        "operational_date": ["01/01/2023", "01/01/2024"],
        "data_source": ["test", "test"],
    })
    monkeypatch.setattr(thermal.pd, "read_csv", lambda *args, **kwargs: source.copy())

    result = thermal.load_historical_additional_thermal_sites("overlay.csv", 2024)

    assert result["station_name"].tolist() == ["new"]
    assert result["capacity_mw"].tolist() == [287.516]
