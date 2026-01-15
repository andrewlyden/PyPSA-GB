"""Utilities for interactive network mapping in the tutorial notebooks."""

from pyproj import Transformer


def prepare_map_network(network):
    """Return a copy of the network with WGS84 coordinates for mapping.

    The original OSGB36 coordinates are preserved in ``x_osgb`` and ``y_osgb``.
    """
    map_network = network.copy()
    map_network.buses["x_osgb"] = map_network.buses["x"]
    map_network.buses["y_osgb"] = map_network.buses["y"]

    x_range = float(map_network.buses["x_osgb"].max() - map_network.buses["x_osgb"].min())
    if x_range > 1000:  # OSGB36 networks span hundreds of kilometers
        transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(
            map_network.buses["x_osgb"].to_numpy(),
            map_network.buses["y_osgb"].to_numpy(),
        )
        map_network.buses["x"] = lon
        map_network.buses["y"] = lat

    return map_network


def explore_network_map(network, *, log_ranges=True, **kwargs):
    """Plot an interactive map using PyPSA's ``plot.explore`` with safe coordinates."""
    map_network = prepare_map_network(network)

    kwargs.setdefault("map_style", "light")
    kwargs.setdefault("tooltip", True)
    kwargs.setdefault("bus_size", 50)
    kwargs.setdefault("bus_size_factor", 2.0)
    kwargs.setdefault("branch_width_factor", 2.0)

    if log_ranges:
        print("lon range:", float(map_network.buses.x.min()), float(map_network.buses.x.max()))
        print("lat range:", float(map_network.buses.y.min()), float(map_network.buses.y.max()))

    return map_network.plot.explore(**kwargs)
