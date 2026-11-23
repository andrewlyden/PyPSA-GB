import pandas as pd
import pypsa
from pathlib import Path

from scripts.generators.integrate_thermal_generators import (
    add_roc_evidenced_dukes_hydro,
    add_thermal_generators,
    standardize_repd_for_merge,
)
from scripts.generators.station_unit_splits import apply_historical_station_unit_splits


def test_repd_standardisation_retains_site_name_as_station_name():
    sites = pd.DataFrame({
        "site_name": ["Example AD plant"],
        "source_site_name": ["Example AD plant"],
        "repd_ref_id": ["REPD-123"],
        "capacity_mw": [2.0],
        "fuel_type": ["biogas"],
    })

    result = standardize_repd_for_merge(sites)

    assert result["station_name"].item() == "Example AD plant"
    assert result["repd_ref_id"].item() == "REPD-123"


def test_repd_generators_use_real_names_and_keep_duplicate_sites():
    network = pypsa.Network()
    network.add("Bus", "test_bus")
    sites = pd.DataFrame({
        "station_name": ["Example AD plant", "Example AD plant"],
        "source_site_name": ["Example AD plant", "Example AD plant"],
        "repd_ref_id": ["REPD-123", "REPD-456"],
        "capacity_mw": [2.0, 3.0],
        "fuel_type": ["biogas", "biogas"],
        "bus": ["test_bus", "test_bus"],
        "data_source": ["REPD", "REPD"],
        "ro_banding": [0.5, 0.5],
    })

    add_thermal_generators(network, sites, solve_mode="LP")

    assert set(network.generators.index) == {
        "Example AD plant",
        "Example AD plant__REPD_REPD-456",
    }
    assert network.generators.at["Example AD plant", "source_site_name"] == "Example AD plant"
    assert network.generators.at["Example AD plant", "repd_ref_id"] == "REPD-123"
    assert network.generators.p_nom.sum() == 5.0


def test_roc_evidenced_dukes_hydro_keeps_station_reference():
    supplement_path = (
        Path(__file__).parents[2]
        / "data/market/ofgem_renewables_obligation/dukes_2020_roc_hydro_supplement.csv"
    )

    network = pypsa.Network()
    network.add("Bus", "test_bus")
    thermal = add_roc_evidenced_dukes_hydro(
        pd.DataFrame(), network, supplement_path, logger=__import__("logging").getLogger(__name__)
    )
    add_thermal_generators(network, thermal, solve_mode="LP")

    assert len(network.generators) == 12
    assert set(network.generators["support_type"]) == {"ROC"}
    assert "R00022SESC" in set(network.generators["ro_accreditation_reference"])


def test_station_unit_split_replaces_source_and_retains_profile():
    network = pypsa.Network()
    network.set_snapshots(pd.date_range("2020-01-01", periods=2, freq="h"))
    network.add("Bus", "test_bus")
    network.add(
        "Generator", "Clyde_Wind_Farm", bus="test_bus", carrier="wind_onshore",
        p_nom=350.0, support_type="ROC", source_site_name="Clyde Wind Farm",
    )
    network.generators_t.p_max_pu["Clyde_Wind_Farm"] = [0.2, 0.8]
    split_file = (
        Path(__file__).parents[2]
        / "data/market/ofgem_renewables_obligation/historical_station_unit_splits_2020.csv"
    )

    apply_historical_station_unit_splits(
        network, split_file, logger=__import__("logging").getLogger(__name__)
    )

    assert set(network.generators.index) == {"Clyde Windfarm (North)", "Clyde Windfarm (Central)"}
    assert network.generators.p_nom.sum() == 393.6
    assert set(network.generators["ro_accreditation_reference"]) == {"R00187SQSC", "R00173SQSC"}
    assert network.generators.at["Clyde Windfarm (North)", "source_generator_name"] == "Clyde_Wind_Farm"
    assert network.generators_t.p_max_pu["Clyde Windfarm (North)"].tolist() == [0.2, 0.8]
