import pandas as pd
import pytest

from scripts.consumer_bill.build_domestic_pes_weights import build_domestic_pes_weights


def test_domestic_weights_use_official_consumption_not_model_load():
    domestic = pd.DataFrame({
        "Year": [2020, 2020], "LA_Code": ["A", "B"],
        "Domestic_consumption_GWh": [100.0, 300.0],
        "Domestic_meters_thouasands": [10.0, 30.0],
    })
    centroids = pd.DataFrame({"LA_Code": ["A", "B"], "lat": [56.0, 51.0], "lon": [-3.0, -1.0]})
    lookup = pd.DataFrame({
        "gsp_name": ["N_1", "S_1"], "gsp_lat": [56.1, 51.1], "gsp_lon": [-3.1, -1.1],
        "pes_id": [22, 11], "pes_name": ["_P", "_B"],
    })

    weights, summary = build_domestic_pes_weights(domestic, centroids, lookup, 2020)

    assert weights.set_index("pes_id").loc[22, "domestic_consumption_weight"] == pytest.approx(0.25)
    assert weights.set_index("pes_id").loc[22, "ofgem_region"] == "Southern Scotland"
    assert summary["unmapped_local_authority_count"] == 0
