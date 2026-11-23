"""Tests for the fixed low-resolution ETYS clustering used in bill backtests."""

from pathlib import Path
import sys

import pandas as pd
import pypsa


sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "network_clustering"))
from consumer_bill_14_zone import GB_ZONES, cluster_buses


def test_14_zone_map_preserves_external_and_non_ac_buses():
    network = pypsa.Network()
    network.add("Bus", "north_scotland", carrier="AC", country="GB", lon=-3.0, lat=58.0)
    network.add("Bus", "london", carrier="AC", country="GB", lon=-0.1, lat=51.5)
    network.add("Bus", "france", carrier="AC", country="France", lon=1.0, lat=50.9)
    network.add("Bus", "hydrogen", carrier="H2", country="GB", lon=-0.1, lat=51.5)

    busmap = cluster_buses(network)

    assert busmap["north_scotland"] == "north_scotland"
    assert busmap["london"] == "london"
    assert busmap["france"] == "external_France_france"
    assert busmap["hydrogen"] == "H2_hydrogen"
    assert set(busmap.loc[["north_scotland", "london"]]).issubset(GB_ZONES)


def test_14_zone_map_assigns_coordinate_free_gb_bus_without_fallback_cluster():
    network = pypsa.Network()
    network.add("Bus", "located", carrier="AC", country="GB", lon=-2.0, lat=52.5)
    network.add("Bus", "missing_coordinates", carrier="AC", country="GB")

    busmap = cluster_buses(network)

    assert busmap.notna().all()
    assert busmap["missing_coordinates"] in GB_ZONES
