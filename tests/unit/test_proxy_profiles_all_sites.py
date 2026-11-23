"""Regional profiles for FES capacity blocks from every site profile (proxy_profile_members: all_sites; issue #51)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "config"))

from scripts.generators.integrate_renewable_generators import (  # noqa: E402
    profile_column_capacities,
    profile_column_locations,
)
from scripts.generators.proxy_profiles import (  # noqa: E402
    NATIONAL_KEY,
    area_keys,
    build_regional_proxy_profiles,
    proxy_profile_settings,
    regions_of_points,
    site_members,
)


def test_settings_default_to_matched_units_and_refuse_unknown_members():
    assert proxy_profile_settings({})["members"] == "matched_units"
    assert proxy_profile_settings({"proxy_profile_members": "all_sites"})["members"] == "all_sites"
    with pytest.raises(ValueError, match="proxy_profile_members"):
        proxy_profile_settings({"proxy_profile_members": "everything"})


def test_points_take_the_region_of_their_gsp_polygon():
    lat = [55.95, 51.50, 52.48, 58.20]   # Edinburgh, London, Birmingham, Moray Firth (offshore: nearest polygon)
    lon = [-3.19, -0.12, -1.90, -2.80]
    settings = proxy_profile_settings({})
    regions = regions_of_points(lat, lon, str(ROOT / settings["region_polygon_file"]),
                                str(ROOT / settings["region_group_file"]))
    assert list(regions) == ["South Scotland", "South East", "Midlands", "North Scotland"]


def test_a_block_takes_its_own_regions_sites():
    hours = pd.RangeIndex(4)
    profiles = pd.DataFrame({
        "North 1": [10.0, 0.0, 10.0, 0.0], "North 2": [20.0, 0.0, 20.0, 0.0], "North 3": [30.0, 0.0, 30.0, 0.0],
        "South 1": [0.0, 10.0, 0.0, 10.0], "South 2": [0.0, 20.0, 0.0, 20.0], "South 3": [0.0, 30.0, 0.0, 30.0],
    }, index=hours)
    locations = pd.DataFrame({"lat": [57.0] * 3 + [51.0] * 3, "lon": [-4.0] * 6,
                              "capacity_mw": [100.0, 200.0, 300.0] * 2}, index=profiles.columns)
    regions = pd.Series(["N"] * 3 + ["S"] * 3, index=profiles.columns)
    members = site_members("wind_onshore", locations, profiles.columns, regions)
    assert len(members) == 6
    proxies = pd.DataFrame({"keys": [area_keys({"cp30_region": "N"}, "wind_onshore"),
                                     area_keys({"cp30_region": "S"}, "wind_onshore")],
                            "weight": [50.0, 50.0]}, index=[0, 1])
    capacities = locations["capacity_mw"].to_dict()
    national = profiles.sum(axis=1) / 1200.0
    settings = {**proxy_profile_settings({"proxy_profile_basis": "regional", "proxy_profile_members": "all_sites"}),
                "min_mw": 100.0, "preserve_carrier_energy": False}
    built, assignments, _, _ = build_regional_proxy_profiles(
        "wind_onshore", profiles, national, members, proxies, capacities, settings)
    np.testing.assert_allclose(built[0].to_numpy(), [0.1, 0.0, 0.1, 0.0])
    np.testing.assert_allclose(built[1].to_numpy(), [0.0, 0.1, 0.0, 0.1])
    assert set(assignments["group"]) == {"N", "S"}


def test_a_thin_region_falls_back_to_gb():
    profiles = pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 1.0]})
    locations = pd.DataFrame({"lat": [57.0, 51.0], "lon": [-4.0, -4.0], "capacity_mw": [1.0, 1.0]}, index=["A", "B"])
    members = site_members("solar_pv", locations, profiles.columns, pd.Series({"A": "N", "B": "S"}))
    proxies = pd.DataFrame({"keys": [area_keys({"cp30_region": "N"}, "solar_pv")], "weight": [1.0]}, index=[7])
    settings = proxy_profile_settings({"proxy_profile_basis": "regional", "proxy_profile_members": "all_sites"})
    _, assignments, _, _ = build_regional_proxy_profiles(
        "solar_pv", profiles, profiles.mean(axis=1), members, proxies, None, settings)
    assert (assignments.iloc[0]["level"], assignments.iloc[0]["group"]) == NATIONAL_KEY


def test_locations_cover_the_same_columns_as_capacities():
    base = pd.DataFrame({"site_name": ["Alpha", "Beta", "Beta", "Gamma"], "capacity_mw": [10.0, 20.0, 25.0, 0.0],
                         "lat": [56.0, 52.0, 52.5, 51.0], "lon": [-3.0, -1.0, -1.5, 0.0]})
    capacities = profile_column_capacities(base)
    locations = profile_column_locations(base)
    assert capacities == {"Alpha": 10.0, "Beta": 25.0}
    assert sorted(locations.index) == sorted(capacities)
    assert locations.loc["Beta", "lat"] == 52.5   # a later duplicate name wins, as for the capacities


def test_fes_scenarios_use_all_sites_and_cp30_keeps_its_matched_units():
    from config_loader import load_layer

    gsp_block = load_layer("method", "gsp_block")["renewables"]
    brownfield = load_layer("method", "brownfield")["renewables"]
    assert gsp_block["proxy_profile_basis"] == "regional"
    assert gsp_block["proxy_profile_members"] == "all_sites"
    assert "proxy_profile_members" not in brownfield   # CP30: matched_units, the code default
