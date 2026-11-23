from pathlib import Path

import pandas as pd
import pytest

from scripts.CP30.generation import load_cp30_generation_capacities, scale_proxy_to_cp30_targets


CP30_ES1 = Path("resources/CP30/CP30_ES1.csv")


@pytest.mark.skipif(not CP30_ES1.exists(), reason="CP30 ES1 resource not available")
def test_cp30_renewable_capacity_targets_match_es1_headlines():
    targets = load_cp30_generation_capacities(
        CP30_ES1,
        "Further Flex and Renewables",
        2030,
        groups=["renewable"],
    )
    by_carrier = targets.set_index("carrier")["capacity_mw"]

    assert by_carrier["wind_offshore"] == pytest.approx(50649.09)
    assert by_carrier["solar_pv"] == pytest.approx(47353.98)
    assert by_carrier["wind_onshore"] == pytest.approx(27326.37)
    assert by_carrier["large_hydro"] == pytest.approx(1897.28)
    assert by_carrier["marine"] == pytest.approx(254.36)


@pytest.mark.skipif(not CP30_ES1.exists(), reason="CP30 ES1 resource not available")
def test_cp30_dispatchable_capacity_targets_include_ccs_and_hydrogen():
    targets = load_cp30_generation_capacities(
        CP30_ES1,
        "New Dispatch",
        2030,
        groups=["thermal"],
    )
    by_carrier = targets.set_index("carrier")["capacity_mw"]

    # CP30 "Gas" is one line split later into CCGT, OCGT and gas engines.
    assert by_carrier["gas"] == pytest.approx(31253.35)
    assert tuple(targets.set_index("carrier").loc["gas", "split_carriers"]) == ("CCGT", "OCGT", "gas_engine")
    assert by_carrier["CHP"] == pytest.approx(3746.65)
    assert by_carrier["nuclear"] == pytest.approx(4130.0)
    assert by_carrier["CCS Biomass"] == pytest.approx(900.0)
    assert by_carrier["CCS Gas"] == pytest.approx(1410.0)
    assert by_carrier["Hydrogen"] == pytest.approx(1276.10)


@pytest.mark.skipif(not CP30_ES1.exists(), reason="CP30 ES1 resource not available")
def test_cp30_generation_targets_exclude_storage_and_interconnectors():
    all_generation = load_cp30_generation_capacities(
        CP30_ES1,
        "Further Flex and Renewables",
        2030,
    )

    assert "Battery" not in set(all_generation["carrier"])
    assert "LDES" not in set(all_generation["carrier"])
    assert "Interconnectors" not in set(all_generation["carrier"])


def test_scale_proxy_to_cp30_targets_preserves_capacity():
    proxy = pd.DataFrame(
        {
            "technology": ["solar_pv", "solar_pv", "wind_onshore"],
            "capacity_mw": [25.0, 75.0, 40.0],
            "bus": ["a", "b", "c"],
            "site_name": ["s1", "s2", "w1"],
        }
    )
    targets = pd.DataFrame(
        {
            "carrier": ["solar_pv", "wind_onshore"],
            "group": ["renewable", "renewable"],
            "capacity_mw": [1000.0, 500.0],
            "pathway": ["Example", "Example"],
            "modelled_year": [2030, 2030],
            "proxy_carriers": [("solar_pv",), ("wind_onshore",)],
            "cp30_type": ["Solar", "Onshore Wind"],
            "cp30_subtype": ["Solar PV", "Onshore Wind"],
        }
    )

    scaled = scale_proxy_to_cp30_targets(
        proxy,
        targets,
        proxy_carrier_col="technology",
        output_carrier_col="technology",
    )
    by_carrier = scaled.groupby("technology")["capacity_mw"].sum()

    assert by_carrier["solar_pv"] == pytest.approx(1000.0)
    assert by_carrier["wind_onshore"] == pytest.approx(500.0)
    assert scaled.loc[scaled["bus"] == "a", "capacity_mw"].iloc[0] == pytest.approx(250.0)
    assert set(scaled["data_source"]) == {"CP30"}


def _target(carrier, capacity, proxy, split=()):
    return {
        "carrier": carrier, "group": "thermal", "capacity_mw": capacity, "pathway": "Example",
        "modelled_year": 2030, "proxy_carriers": proxy, "split_carriers": split,
        "cp30_type": "Gas", "cp30_subtype": carrier,
    }


def test_split_target_divides_on_proxy_shares():
    proxy = pd.DataFrame({
        "fuel_type": ["CCGT", "CCGT", "OCGT", "gas_engine", "CHP"],
        "capacity_mw": [60.0, 15.0, 15.0, 10.0, 50.0],
        "gsp": ["a", "b", "c", "d", "e"],
    })
    targets = pd.DataFrame([_target("gas", 1000.0, ("CCGT",), ("CCGT", "OCGT", "gas_engine"))])
    scaled = scale_proxy_to_cp30_targets(proxy, targets, "fuel_type", "fuel_type")
    by_carrier = scaled.groupby("fuel_type")["capacity_mw"].sum()
    assert by_carrier["CCGT"] == pytest.approx(750.0)
    assert by_carrier["OCGT"] == pytest.approx(150.0)
    assert by_carrier["gas_engine"] == pytest.approx(100.0)
    assert "CHP" not in by_carrier
    assert by_carrier.sum() == pytest.approx(1000.0)


def test_proxy_carriers_are_an_ordered_fallback_not_a_pool():
    proxy = pd.DataFrame({
        "fuel_type": ["CHP", "CCGT", "oil", "OCGT"],
        "capacity_mw": [10.0, 90.0, 5.0, 95.0],
        "gsp": ["chp", "ccgt", "oil", "ocgt"],
    })
    targets = pd.DataFrame([
        _target("CHP", 100.0, ("CHP", "CCGT")),
        _target("oil", 20.0, ("oil", "OCGT", "CCGT")),
        _target("Hydrogen", 30.0, ("Hydrogen", "H2", "CCGT")),
    ])
    scaled = scale_proxy_to_cp30_targets(proxy, targets, "fuel_type", "fuel_type")
    # CHP and oil sit only on their own rows; hydrogen, with none, falls back to CCGT.
    assert set(scaled.loc[scaled["fuel_type"].eq("CHP"), "gsp"]) == {"chp"}
    assert set(scaled.loc[scaled["fuel_type"].eq("oil"), "gsp"]) == {"oil"}
    assert set(scaled.loc[scaled["fuel_type"].eq("Hydrogen"), "gsp"]) == {"ccgt"}
