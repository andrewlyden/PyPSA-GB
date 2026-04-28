import logging
from pathlib import Path

import pandas as pd

from scripts.generators.build_bmu_mapping import (
    _fuel_to_carrier_candidates,
    _load_power_station_dictionary,
    _normalise_station_name,
    _select_generator_match,
)


def test_load_power_station_dictionary_normalises_columns():
    csv_path = Path("tests/.tmp_power_station_dictionary_bmu_crosswalk.csv")
    pd.DataFrame(
        [
            {
                "Settlement BMU ID": "E_CORB-1",
                "Common Name": "Corby Power Station",
                "Fuel Type": "Natural Gas",
                "Plant Type": "CCGT",
                "Installed Capacity (MW)": 401.0,
            }
        ]
    ).to_csv(csv_path, index=False)

    df = _load_power_station_dictionary(str(csv_path), logging.getLogger("test"))

    assert list(df["bmu_id"]) == ["E_CORB-1"]
    assert list(df["station_name"]) == ["Corby Power Station"]
    assert list(df["station_name_norm"]) == ["corby"]
    assert float(df.iloc[0]["installed_capacity_mw"]) == 401.0


def test_fuel_to_carrier_candidates_maps_dictionary_hints():
    carriers = _fuel_to_carrier_candidates("Natural Gas", "CCGT")
    assert "CCGT" in carriers

    carriers = _fuel_to_carrier_candidates("Wind", "Wind Offshore")
    assert "wind_offshore" in carriers


def test_select_generator_match_prefers_carrier_and_capacity_for_ambiguous_station():
    lookup_df = pd.DataFrame(
        [
            {
                "name": "West Burton",
                "name_norm": _normalise_station_name("West Burton"),
                "carrier": "coal",
                "p_nom": 2000.0,
                "bus": "WBUR42",
                "component_type": "generator",
                "data_source": "DUKES",
            },
            {
                "name": "West Burton CCGT",
                "name_norm": _normalise_station_name("West Burton CCGT"),
                "carrier": "CCGT",
                "p_nom": 1332.0,
                "bus": "WBUR42",
                "component_type": "generator",
                "data_source": "DUKES",
            },
            {
                "name": "West Burton GT",
                "name_norm": _normalise_station_name("West Burton GT"),
                "carrier": "OCGT",
                "p_nom": 40.0,
                "bus": "BRFO41",
                "component_type": "generator",
                "data_source": "DUKES",
            },
        ]
    )

    matched, method = _select_generator_match(
        station_name="West Burton",
        lookup_df=lookup_df,
        expected_carriers={"CCGT"},
        expected_bus="WBUR42",
        expected_capacity_mw=1332.0,
    )

    assert matched == "West Burton CCGT"
    assert method == "partial"


def test_select_generator_match_uses_explicit_dictionary_generator_name():
    lookup_df = pd.DataFrame(
        [
            {
                "name": "Grain GT",
                "name_norm": _normalise_station_name("Grain GT"),
                "carrier": "OCGT",
                "p_nom": 56.0,
                "bus": "FLEE11",
                "component_type": "generator",
                "data_source": "DUKES",
            },
            {
                "name": "Grain CHP*",
                "name_norm": _normalise_station_name("Grain CHP*"),
                "carrier": "CCGT",
                "p_nom": 1517.0,
                "bus": "GRAI41",
                "component_type": "generator",
                "data_source": "DUKES",
            },
        ]
    )

    matched, method = _select_generator_match(
        station_name="Grain",
        lookup_df=lookup_df,
        explicit_generator_name="Grain CHP*",
        expected_carriers={"CCGT"},
        expected_bus="GRAI41",
        expected_capacity_mw=1517.0,
    )

    assert matched == "Grain CHP*"
    assert method == "power_station_dictionary"
