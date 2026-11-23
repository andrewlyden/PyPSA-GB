"""ESPENI embedded wind is placed where the small wind farms it stands for are (18 Nov 2026), not by demand."""

import pandas as pd
import pypsa

from scripts.generators.integrate_renewable_generators import _embedded_site_bus_share


def _network():
    n = pypsa.Network()
    # OSGB36 metres: a load bus in north Scotland, one in southern England, one without load
    n.add("Bus", "north", x=270000.0, y=850000.0)
    n.add("Bus", "south", x=530000.0, y=180000.0)
    n.add("Bus", "nowhere", x=275000.0, y=855000.0)
    return n


def test_capacity_goes_to_the_nearest_load_bus():
    sites = pd.DataFrame({"lat": [57.5, 57.6, 51.5], "lon": [-4.3, -4.2, -0.1], "capacity_mw": [6.0, 2.0, 2.0]})
    share = _embedded_site_bus_share(_network(), sites, ["north", "south"])
    assert share.round(3).to_dict() == {"north": 0.8, "south": 0.2}


def test_no_usable_site_keeps_the_fallback():
    sites = pd.DataFrame({"lat": [None], "lon": [None], "capacity_mw": [5.0]})
    assert _embedded_site_bus_share(_network(), sites, ["north", "south"]) is None
