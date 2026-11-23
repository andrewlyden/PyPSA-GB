from pathlib import Path

import pandas as pd
import pytest

from scripts.CP30.storage_interconnectors import (
    load_cp30_interconnector_capacity_mw,
    load_cp30_storage_targets,
    scale_storage_proxy_to_cp30_targets,
)
from config.config_loader import load_defaults


CP30_ES1 = Path("resources/CP30/CP30_ES1.csv")


@pytest.mark.skipif(not CP30_ES1.exists(), reason="CP30 ES1 resource not available")
def test_cp30_storage_targets_include_power_energy_and_duration():
    targets = load_cp30_storage_targets(CP30_ES1, "Further Flex and Renewables", 2030)
    by_carrier = targets.set_index("carrier")

    assert by_carrier.loc["Battery", "power_mw"] == pytest.approx(27383.45)
    assert by_carrier.loc["Battery", "energy_mwh"] == pytest.approx(48857.64)
    assert by_carrier.loc["Battery", "duration_h"] == pytest.approx(48857.64 / 27383.45)

    assert by_carrier.loc["LAES", "power_mw"] == pytest.approx(7858.19)
    assert by_carrier.loc["LAES", "energy_mwh"] == pytest.approx(98928.20)
    assert by_carrier.loc["LAES", "duration_h"] == pytest.approx(98928.20 / 7858.19)


@pytest.mark.skipif(not CP30_ES1.exists(), reason="CP30 ES1 resource not available")
def test_cp30_new_dispatch_storage_targets():
    targets = load_cp30_storage_targets(CP30_ES1, "New Dispatch", 2030)
    by_carrier = targets.set_index("carrier")

    assert by_carrier.loc["Battery", "power_mw"] == pytest.approx(22625.37)
    assert by_carrier.loc["Battery", "energy_mwh"] == pytest.approx(40403.94)
    assert by_carrier.loc["LAES", "power_mw"] == pytest.approx(4573.89)
    assert by_carrier.loc["LAES", "energy_mwh"] == pytest.approx(50110.00)


def test_scale_storage_proxy_to_cp30_targets_preserves_power_and_energy():
    proxy = pd.DataFrame(
        {
            "technology": ["Battery", "Domestic Battery", "LAES"],
            "power_mw": [100.0, 100.0, 50.0],
            "energy_mwh": [200.0, 200.0, 300.0],
            "duration_h": [2.0, 2.0, 6.0],
            "site_name": ["b1", "b2", "l1"],
            "lat": [52.0, 53.0, 54.0],
            "lon": [-1.0, -2.0, -3.0],
        }
    )
    targets = pd.DataFrame(
        {
            "carrier": ["Battery", "LAES"],
            "cp30_subtype": ["Battery", "LDES"],
            "power_mw": [1000.0, 300.0],
            "energy_mwh": [1800.0, 3600.0],
            "energy_gwh": [1.8, 3.6],
            "duration_h": [1.8, 12.0],
            "eta_charge": [0.92, 0.60],
            "eta_discharge": [0.92, 0.60],
            "pathway": ["Example", "Example"],
            "modelled_year": [2030, 2030],
            "proxy_technologies": [("Battery", "Domestic Battery"), ("LAES",)],
        }
    )

    scaled = scale_storage_proxy_to_cp30_targets(proxy, targets)
    by_carrier = scaled.groupby("technology").agg({"power_mw": "sum", "energy_mwh": "sum"})

    assert by_carrier.loc["Battery", "power_mw"] == pytest.approx(1000.0)
    assert by_carrier.loc["Battery", "energy_mwh"] == pytest.approx(1800.0)
    assert by_carrier.loc["LAES", "power_mw"] == pytest.approx(300.0)
    assert by_carrier.loc["LAES", "energy_mwh"] == pytest.approx(3600.0)
    assert set(scaled["data_source"]) == {"CP30"}


@pytest.mark.skipif(not CP30_ES1.exists(), reason="CP30 ES1 resource not available")
def test_cp30_interconnector_target_is_es1_total():
    assert load_cp30_interconnector_capacity_mw(CP30_ES1, "Further Flex and Renewables", 2030) == pytest.approx(12450.0)
    assert load_cp30_interconnector_capacity_mw(CP30_ES1, "New Dispatch", 2030) == pytest.approx(12450.0)


ROUND_TRIPS = {"Battery": 0.8464, "LAES": 0.55, "LDES": 0.70}


@pytest.mark.skipif(not CP30_ES1.exists(), reason="CP30 ES1 resource not available")
def test_cp30_ldes_takes_the_configured_round_trip_not_a_hard_coded_leg():
    targets = load_cp30_storage_targets(
        CP30_ES1, "Further Flex and Renewables", 2030, round_trip_efficiencies=ROUND_TRIPS
    ).set_index("cp30_subtype")

    # The subtype key wins over the carrier key: LDES 0.70, not LAES 0.55,
    # and certainly not the 0.36 that 0.60 per leg gave.
    assert targets.loc["LDES", "round_trip_efficiency"] == pytest.approx(0.70)
    assert targets.loc["LDES", "eta_charge"] == pytest.approx(0.70 ** 0.5)
    assert targets.loc["LDES", "eta_charge"] * targets.loc["LDES", "eta_discharge"] == pytest.approx(0.70)
    assert targets.loc["Battery", "eta_charge"] == pytest.approx(0.92)
    # CP30's own generation / consumption, reported alongside
    assert targets.loc["LDES", "cp30_implied_round_trip"] == pytest.approx(10441.51 / 14848.69)


def test_cp30_storage_without_a_configured_round_trip_is_refused():
    from scripts.CP30.storage_interconnectors import cp30_round_trip_efficiency

    assert cp30_round_trip_efficiency("LDES", "LAES", {"LAES": 0.55}) == pytest.approx(0.55)
    with pytest.raises(ValueError, match=r"storage\.round_trip_efficiencies"):
        cp30_round_trip_efficiency("LDES", "LAES", {"Battery": 0.8464})
    with pytest.raises(ValueError, match="must be in"):
        cp30_round_trip_efficiency("LDES", "LAES", {"LDES": 1.4})


def test_scaling_without_target_efficiency_keeps_the_proxy_efficiency():
    proxy = pd.DataFrame({
        "technology": ["LAES"], "power_mw": [50.0], "duration_h": [6.0],
        "eta_charge": [0.55 ** 0.5], "eta_discharge": [0.55 ** 0.5],
    })
    targets = pd.DataFrame({
        "carrier": ["LAES"], "cp30_subtype": ["LDES"], "power_mw": [300.0],
        "energy_mwh": [3600.0], "duration_h": [12.0], "pathway": ["Example"],
        "proxy_technologies": [("LAES",)],
    })
    scaled = scale_storage_proxy_to_cp30_targets(proxy, targets)
    assert scaled["eta_charge"].iloc[0] == pytest.approx(0.55 ** 0.5)

    targets["eta_charge"] = targets["eta_discharge"] = 0.70 ** 0.5
    scaled = scale_storage_proxy_to_cp30_targets(proxy, targets)
    assert (scaled["eta_charge"] * scaled["eta_discharge"]).iloc[0] == pytest.approx(0.70)


# ----------------------------------------------------------------------------- existing pumped hydro (fix S3)

EXISTING = [
    {"name": "Dinorwig", "site_code": "DINO", "lat": 53.12, "lon": -4.11, "power_mw": 1728, "energy_mwh": 9100},
    {"name": "Cruachan", "site_code": "CRUA", "lat": 56.41, "lon": -5.11, "power_mw": 440, "energy_mwh": 7100},
    {"name": "Foyers", "site_code": "FOYE", "lat": 57.25, "lon": -4.49, "power_mw": 300, "energy_mwh": 6400,
     "enabled": False},
]


def _ldes_proxy():
    return pd.DataFrame({
        "technology": ["LAES"] * 4,
        "power_mw": [50.0, 30.0, 10.0, 10.0],
        "duration_h": [6.0] * 4,
        "site_name": ["FES_Direct_LAES_Dinorwig_2030", "FES_Direct_LAES_Cruachan_2030", "FES_LAES_A_2030", "FES_LAES_B_2030"],
        "lat": [53.1, 56.4, 52.0, 51.0], "lon": [-4.1, -5.1, -1.0, 0.0],
    })


def _ldes_target(power=7858.19, energy=98928.20):
    return pd.DataFrame({
        "carrier": ["LAES"], "cp30_subtype": ["LDES"], "power_mw": [power], "energy_mwh": [energy],
        "duration_h": [energy / power], "eta_charge": [0.70 ** 0.5], "eta_discharge": [0.70 ** 0.5],
        "pathway": ["Example"], "proxy_technologies": [("LAES",)],
    })


def test_existing_pumped_hydro_keeps_real_size_and_cp30_ldes_total_is_exact():
    scaled = scale_storage_proxy_to_cp30_targets(_ldes_proxy(), _ldes_target(), existing_sites=EXISTING)
    assert scaled["power_mw"].sum() == pytest.approx(7858.19)
    assert scaled["energy_mwh"].sum() == pytest.approx(98928.20)
    existing = scaled[scaled["existing_site"].notna()].set_index("existing_site")
    assert set(existing.index) == {"Dinorwig", "Cruachan"}  # Foyers disabled
    assert existing.loc["Dinorwig", ["power_mw", "energy_mwh"]].tolist() == pytest.approx([1728.0, 9100.0])
    assert existing.loc["Cruachan", "duration_h"] == pytest.approx(7100.0 / 440.0)
    assert existing.loc["Dinorwig", "site_code"] == "DINO"
    assert existing.loc["Dinorwig", "site_name"] == "CP30_LAES_existing_Dinorwig"
    # The existing sites' own REPD proxy rows are not also given new build.
    new = scaled[scaled["existing_site"].isna()]
    assert len(new) == 2
    assert new["power_mw"].sum() == pytest.approx(7858.19 - 2168.0)
    assert new["duration_h"].unique() == pytest.approx([(98928.20 - 16200.0) / (7858.19 - 2168.0)])
    assert (scaled["eta_charge"] * scaled["eta_discharge"]).unique() == pytest.approx([0.70])


def test_existing_sites_that_fill_the_target_are_refused():
    with pytest.raises(ValueError, match="leave no new build"):
        scale_storage_proxy_to_cp30_targets(_ldes_proxy(), _ldes_target(power=2000.0, energy=50000.0),
                                            existing_sites=EXISTING)


def test_existing_site_needs_a_four_character_site_code():
    from scripts.CP30.storage_interconnectors import parse_existing_storage_sites

    with pytest.raises(ValueError, match="four characters"):
        parse_existing_storage_sites([dict(EXISTING[0], site_code="DINO41")])
    assert [s["name"] for s in parse_existing_storage_sites(EXISTING)] == ["Dinorwig", "Cruachan"]


def test_configured_existing_sites_match_public_ownership_register_sites():
    import yaml

    defaults = load_defaults()
    from scripts.CP30.storage_interconnectors import parse_existing_storage_sites

    network_sites = {s["name"]: s for s in parse_existing_storage_sites(defaults["renewables"]["hydro"]["pumped_hydro"]["existing_sites"])}
    register_sites = {s["name"]: s for s in defaults["public_ownership"]["existing_storage_sites"]}
    assert set(network_sites) == set(register_sites)
    for name, site in register_sites.items():
        assert float(site["existing_mw"]) == pytest.approx(network_sites[name]["power_mw"])
        assert site["bus"][:4] == network_sites[name]["site_code"]
