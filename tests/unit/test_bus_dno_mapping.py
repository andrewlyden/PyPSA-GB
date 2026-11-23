import pandas as pd
import pytest

from scripts.consumer_bill.build_bus_dno_mapping import build_bus_dno_mapping


def test_bus_mapping_uses_nearest_gsp_and_preserves_pes_area():
    buses = pd.DataFrame({"bus": ["north", "south"], "lat": [56.0, 51.0], "lon": [-3.0, -1.0]})
    lookup = pd.DataFrame({
        "gsp_name": ["N_1", "S_1"], "gsp_lat": [56.1, 51.1], "gsp_lon": [-3.1, -1.1],
        "pes_id": [22, 11], "pes_name": ["_P", "_B"],
    })

    mapping = build_bus_dno_mapping(buses, lookup)

    assert mapping.set_index("bus").loc["north", "pes_id"] == 22
    assert mapping.set_index("bus").loc["south", "nearest_gsp_name"] == "S_1"


def test_bus_mapping_rejects_ambiguous_gsp_to_pes_lookup():
    buses = pd.DataFrame({"bus": ["bus"], "lat": [56.0], "lon": [-3.0]})
    lookup = pd.DataFrame({
        "gsp_name": ["N_1", "N_1"], "gsp_lat": [56.1, 56.1], "gsp_lon": [-3.1, -3.1],
        "pes_id": [22, 23], "pes_name": ["_P", "_Q"],
    })

    with pytest.raises(ValueError, match="multiple PES"):
        build_bus_dno_mapping(buses, lookup)
