"""Where FES GSP-level renewables go (user decisions, 27 Oct 2026).

- Rows are located by GSP id first: names drift between editions and tables ("Bridgwater" and
  "Bridgewater", "Clydes Mill" and "Clyde's Mill").
- Each edition's own committed GSP table, with NESO's published point by GSP id.
- On the FES long-range chain, a row goes on its GSP's demand node (the 2030 rule).
- The island GSPs are placed at declared positions; any other unplaced GSP stops the build.
"""

import pandas as pd
import pypsa
import pytest

from scripts.FES.gsp_metadata import substation_points
from scripts.generators.integrate_renewable_generators import (
    DECLARED_GSP_POINTS_FILE,
    _locate_gsp,
)
from scripts.utilities.gsp_nodes import GspDemandNodes


MAPPING = {
    "Bridgewater": {"bus": "BRWA_POINT", "lat": 51.1, "lon": -3.0, "gsp_id": "BRWA1"},
    "Clyde’s Mill": {"bus": "CLYM_POINT", "lat": 55.8, "lon": -4.1, "gsp_id": "CLYM"},
    "Eday (Orkney)": {"bus": "ORKNEY", "lat": 59.19, "lon": -2.77, "gsp_id": ""},
}


def test_gsp_id_wins_over_a_drifted_name():
    assert _locate_gsp("Bridgwater", "BRWA1", MAPPING)["bus"] == "BRWA_POINT"
    assert _locate_gsp("Clydes Mill", "CLYM", MAPPING)["bus"] == "CLYM_POINT"


def test_rows_without_an_id_fall_back_to_the_name():
    assert _locate_gsp("Eday (Orkney)", None, MAPPING)["bus"] == "ORKNEY"
    assert _locate_gsp("Eday (Orkney)", float("nan"), MAPPING)["bus"] == "ORKNEY"


def test_declared_points_cover_the_island_gsps():
    declared = pd.read_csv(DECLARED_GSP_POINTS_FILE, encoding="utf-8-sig")
    assert {"Eday (Orkney)", "Harris"} <= set(declared["Name"])
    assert declared["Latitude"].between(49, 61).all() and declared["Longitude"].between(-9, 2).all()
    assert declared["source"].str.len().gt(0).all()


def _network() -> pypsa.Network:
    n = pypsa.Network()
    n.add("Bus", ["AAAA11", "AAAA41", "BBBB11"], x=[400000, 400100, 450000], y=[300000, 300100, 350000])
    n.add("Load", ["L1", "L2"], bus=["AAAA11", "BBBB11"], p_set=1.0)
    return n


def test_a_gsp_goes_on_its_demand_node():
    dem = pd.DataFrame({"GSP Id": ["AAAA_1", "AAAA_1", "BBBB_1"], "Node Id": ["AAAA11", "BBBB11", "BBBB11"],
                        "Dem as % of GSP": [70.0, 30.0, 100.0]})
    nodes = GspDemandNodes(_network(), dem_per_node=dem)
    assert nodes.node("AAAA_1")[0] == "AAAA11"          # the largest share
    assert nodes.node("BBBB_1")[0] == "BBBB11"
    assert nodes.node("CCCC_1") == (None, "no_dem_per_node_row")
    assert nodes.node(None) == (None, "no_dem_per_node_row")


def test_fes2025_table_carries_neso_points_by_gsp_id():
    t25 = pd.read_csv("data/network/ETYS/fes2025_regional_breakdown_gsp_info.csv", encoding="utf-8-sig")
    t24 = pd.read_csv("data/network/ETYS/fes2024_regional_breakdown_gsp_info.csv", encoding="utf-8-sig")
    t24 = t24.drop_duplicates("GSP ID").set_index("GSP ID")
    shared = t25[t25["GSP ID"].isin(t24.index)]
    assert len(shared) >= 370
    assert (shared["Latitude"].to_numpy() == t24.loc[shared["GSP ID"], "Latitude"].to_numpy()).all()
    assert (shared["Longitude"].to_numpy() == t24.loc[shared["GSP ID"], "Longitude"].to_numpy()).all()
    # Unchanged by a second pass: the committed table is the generator's rule applied.
    again = substation_points(t25, 2025)
    assert (again["Latitude"] == t25["Latitude"]).all() and (again["Longitude"] == t25["Longitude"]).all()


def test_licence_areas_follow_the_gsp_polygons():
    # FES Direct(SHETL/SPTL/NGET) capacity by GSP polygon, not latitude bands (27 Oct 2026).
    from scripts.CP30.spatial_allocation import LicenceAreas

    buses = pd.DataFrame({"x": [266000.0, 266500.0, 340000.0, 260000.0], "y": [845000.0, 665000.0, 556000.0, 730000.0]},
                         index=["INVE1-", "GLAS1-", "CARL1-", "CRUA2Q"])
    areas = LicenceAreas(buses)
    sites = areas.site_licence([57.48, 55.86, 54.89], [-4.22, -4.25, -2.93])
    assert sites.tolist() == ["SHETL", "SPTL", "NGET"]
    assert areas.bus_licence["CRUA2Q"] == "SPTL"          # transmission-site override
    assert areas.nearest_bus(57.0, -4.5, "SHETL") == "INVE1-"
    assert areas.nearest_bus(57.0, -4.5, "SPTL") in {"GLAS1-", "CRUA2Q"}
