import logging
from pathlib import Path

import pandas as pd
import pytest

import scripts.demand.load as demand_load


CP30_ED1 = Path("resources/CP30/CP30_ED1.csv")
CP30_ES1 = Path("resources/CP30/CP30_ES1.csv")
FES_2025 = Path("resources/FES/FES_2025_data.csv")


@pytest.mark.skipif(not CP30_ED1.exists(), reason="CP30 ED1 resource not available")
def test_cp30_demand_uses_ed1_when_present():
    value = demand_load.get_cp30_electricity_demand_gwh(
        "Further Flex and Renewables",
        2030,
        CP30_ED1,
        cp30_es1_path=CP30_ES1,
    )

    assert value == pytest.approx(315162.7497490148)


@pytest.mark.skipif(not CP30_ED1.exists() or not CP30_ES1.exists(), reason="CP30 resources not available")
def test_cp30_new_dispatch_uses_cp30_es1_when_ed1_annual_row_missing():
    value = demand_load.get_cp30_electricity_demand_gwh(
        "New Dispatch",
        2030,
        CP30_ED1,
        cp30_es1_path=CP30_ES1,
    )

    assert value == pytest.approx(315161.95)


def test_scale_gsp_shares_to_annual_total_preserves_cp30_total():
    gsp_demand = pd.DataFrame({"2030": [400.0, 600.0]}, index=["A", "B"])

    scaled, scale_factor = demand_load.scale_gsp_shares_to_annual_total(
        gsp_demand,
        1000.0,
        315000.0,
        "2030",
    )

    assert scale_factor == pytest.approx(315.0)
    assert scaled.loc["A", "2030"] == pytest.approx(126000.0)
    assert scaled.loc["B", "2030"] == pytest.approx(189000.0)
    assert scaled["2030"].sum() == pytest.approx(315000.0)


@pytest.mark.skipif(
    not CP30_ED1.exists() or not CP30_ES1.exists() or not FES_2025.exists(),
    reason="CP30/FES resources not available",
)
def test_get_cp30_demand_data_uses_fes_only_for_spatial_proxy(monkeypatch):
    monkeypatch.setattr(demand_load, "get_network_model", lambda: "Reduced")

    result = demand_load.get_cp30_demand_data(
        "Further Flex and Renewables",
        2030,
        logging.getLogger("test_cp30_demand"),
        CP30_ED1,
        CP30_ES1,
        "Holistic Transition",
        spatial_fes_year=2025,
        fes_data_path=str(FES_2025),
    )

    assert len(result) > 0
    assert result["2030"].sum() == pytest.approx(315162.7497490148)
    assert (result["2030"] >= 0).all()

