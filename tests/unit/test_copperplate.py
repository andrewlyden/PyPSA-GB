"""Tests for the exact copperplate collapse in scripts/market/copperplate.py.

Synthetic network with two synchronous sub-networks and an External bus:

    M1 --line-- M2        mainland (two buses, one AC line, an internal DC link)
    M2 --DC(50)-- I1      island with a cheap generator (export link binds)
    M1 --IC(100)-- External_FR   cheap foreign generator (interconnector binds)

Collapsing the mainland must give one mainland bus, keep the island and both
links, drop the line and the internal link, and reproduce the objective of the
legacy s_nom=1e6 relaxation on the full network.
"""

import logging

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.market.copperplate import (
    MAINLAND_HUB,
    CollapseMapping,
    collapse_mainland,
    expand_results,
    identify_mainland,
)

SNAPSHOTS = pd.date_range("2030-01-01", periods=3, freq="h")


def _build_network() -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots(SNAPSHOTS)
    for bus, country, x, y in (
        ("M1", "GB", 0.0, 0.0),
        ("M2", "GB", 1.0, 0.0),
        ("I1", "GB", 2.0, 2.0),
        ("External_FR", "France", 5.0, -5.0),
    ):
        n.add("Bus", bus, v_nom=400.0, carrier="AC", country=country, x=x, y=y)
    n.add("Line", "M1-M2", bus0="M1", bus1="M2", x=0.1, r=0.01, s_nom=1000.0)
    n.add("Link", "internal_hvdc", bus0="M1", bus1="M2", p_nom=1000.0, p_min_pu=-1.0, efficiency=1.0)
    n.add("Link", "island_export", bus0="I1", bus1="M2", p_nom=50.0, p_min_pu=-1.0, efficiency=1.0)
    n.add("Link", "IC_FR", bus0="External_FR", bus1="M1", p_nom=100.0, p_min_pu=-1.0, efficiency=1.0)
    n.add("Load", "L1", bus="M1", p_set=200.0)
    n.add("Load", "L2", bus="M2", p_set=100.0)
    n.add("Generator", "island_wind", bus="I1", p_nom=100.0, marginal_cost=5.0, carrier="wind_offshore")
    n.add("Generator", "gb_ccgt", bus="M1", p_nom=300.0, marginal_cost=30.0, carrier="CCGT")
    n.add("Generator", "gb_ocgt", bus="M2", p_nom=300.0, marginal_cost=80.0, carrier="OCGT")
    n.add("Generator", "fr_nuclear", bus="External_FR", p_nom=500.0, marginal_cost=20.0, carrier="EU_import")
    n.add(
        "StorageUnit", "gb_battery", bus="M2", p_nom=20.0, max_hours=2.0,
        efficiency_store=0.95, efficiency_dispatch=0.95, marginal_cost=1.0,
        carrier="battery", state_of_charge_initial=20.0,
    )
    return n


def _relaxed(n: pypsa.Network) -> pypsa.Network:
    r = n.copy()
    r.lines["s_nom"] = 1.0e6
    return r


def _solve(n: pypsa.Network) -> float:
    status, condition = n.optimize(solver_name="highs")
    assert status == "ok" and condition == "optimal", (status, condition)
    return float(n.objective)


class TestIdentifyMainland:
    def test_largest_non_external_sub_network_is_the_mainland(self):
        n = _build_network()
        main, n_sub = identify_mainland(n)
        assert main == {"M1", "M2"}
        assert n_sub == 2  # mainland and the island; the External bus is excluded


class TestCollapseMainland:
    def test_topology_after_collapse(self):
        n = _build_network()
        c, mapping = collapse_mainland(n, logging.getLogger("t"))

        assert isinstance(mapping, CollapseMapping)
        assert set(c.buses.index) == {MAINLAND_HUB, "I1", "External_FR"}
        assert len(c.lines) == 0
        assert mapping.removed_lines == ["M1-M2"]
        assert mapping.removed_links == ["internal_hvdc"]
        assert sorted(mapping.mainland_buses) == ["M1", "M2"]
        assert set(c.links.index) == {"island_export", "IC_FR"}
        assert c.links.loc["island_export", "bus1"] == MAINLAND_HUB
        assert c.links.loc["island_export", "bus0"] == "I1"
        assert c.links.loc["IC_FR", "bus1"] == MAINLAND_HUB
        assert c.links.loc["IC_FR", "bus0"] == "External_FR"
        assert float(c.links.loc["island_export", "p_nom"]) == 50.0
        assert set(c.generators.index) == set(n.generators.index)
        assert c.generators.loc["gb_ccgt", "bus"] == MAINLAND_HUB
        assert c.generators.loc["island_wind", "bus"] == "I1"
        assert c.generators.loc["fr_nuclear", "bus"] == "External_FR"
        assert (c.loads["bus"] == MAINLAND_HUB).all()
        assert c.storage_units.loc["gb_battery", "bus"] == MAINLAND_HUB
        assert c.buses.loc[MAINLAND_HUB, "country"] == "GB"
        assert c.buses.loc[MAINLAND_HUB, "v_nom"] == 400.0
        assert c.buses.loc[MAINLAND_HUB, "x"] == pytest.approx(0.5)

    def test_input_network_is_not_modified(self):
        n = _build_network()
        buses_before = list(n.buses.index)
        gen_buses_before = n.generators["bus"].to_dict()
        collapse_mainland(n, logging.getLogger("t"))
        assert list(n.buses.index) == buses_before
        assert n.generators["bus"].to_dict() == gen_buses_before
        assert len(n.lines) == 1 and len(n.links) == 3

    def test_collapsing_twice_is_refused(self):
        n = _build_network()
        c, _ = collapse_mainland(n, logging.getLogger("t"))
        with pytest.raises(ValueError, match="already"):
            collapse_mainland(c, logging.getLogger("t"))


class TestCollapsedSolveMatchesRelaxation:
    def test_objective_equals_relaxed_full_network(self):
        n = _build_network()
        relaxed = _relaxed(n)
        collapsed, _ = collapse_mainland(n, logging.getLogger("t"))
        obj_relaxed = _solve(relaxed)
        obj_collapsed = _solve(collapsed)
        assert obj_collapsed == pytest.approx(obj_relaxed, rel=1e-6)

    def test_binding_links_stay_binding_and_dispatch_matches(self):
        n = _build_network()
        relaxed = _relaxed(n)
        collapsed, _ = collapse_mainland(n, logging.getLogger("t"))
        _solve(relaxed)
        _solve(collapsed)
        # The island's 100 MW of 5 GBP wind is capped by its 50 MW export link,
        # and the 20 GBP French import by the 100 MW interconnector, in both.
        assert np.allclose(collapsed.links_t.p0["island_export"], 50.0, atol=1e-6)
        assert np.allclose(relaxed.links_t.p0["island_export"], 50.0, atol=1e-6)
        assert np.allclose(collapsed.links_t.p0["IC_FR"], 100.0, atol=1e-6)
        assert np.allclose(relaxed.links_t.p0["IC_FR"], 100.0, atol=1e-6)
        for gen in n.generators.index:
            assert np.allclose(
                collapsed.generators_t.p[gen], relaxed.generators_t.p[gen], atol=1e-6
            ), gen
        # One mainland price in both networks
        assert np.allclose(
            collapsed.buses_t.marginal_price[MAINLAND_HUB],
            relaxed.buses_t.marginal_price["M1"], atol=1e-6,
        )
        assert np.allclose(
            relaxed.buses_t.marginal_price["M1"], relaxed.buses_t.marginal_price["M2"], atol=1e-6
        )


class TestExpandResults:
    def test_every_original_bus_price_and_generator_column_is_filled(self):
        n = _build_network()
        collapsed, mapping = collapse_mainland(n, logging.getLogger("t"))
        _solve(collapsed)
        expand_results(n, collapsed, mapping, logging.getLogger("t"))

        prices = n.buses_t.marginal_price
        assert list(prices.columns) == list(n.buses.index)
        assert prices.index.equals(n.snapshots)
        hub = collapsed.buses_t.marginal_price[MAINLAND_HUB]
        assert np.allclose(prices["M1"], hub)
        assert np.allclose(prices["M2"], hub)
        assert np.allclose(prices["I1"], collapsed.buses_t.marginal_price["I1"])
        assert np.allclose(prices["External_FR"], collapsed.buses_t.marginal_price["External_FR"])

        gen_p = n.generators_t.p
        assert list(gen_p.columns) == list(n.generators.index)
        for gen in n.generators.index:
            assert np.allclose(gen_p[gen], collapsed.generators_t.p[gen]), gen

        assert list(n.storage_units_t.p.columns) == ["gb_battery"]
        assert np.allclose(n.storage_units_t.p["gb_battery"], collapsed.storage_units_t.p["gb_battery"])

        links = n.links_t.p0
        assert set(links.columns) == set(n.links.index)
        assert np.allclose(links["internal_hvdc"], 0.0)  # removed by the collapse -> zero column
        assert np.allclose(links["island_export"], collapsed.links_t.p0["island_export"])

        lines = n.lines_t.p0
        assert list(lines.columns) == ["M1-M2"]
        assert np.allclose(lines["M1-M2"], 0.0)

        assert n.objective == pytest.approx(collapsed.objective)

    def test_snapshot_mismatch_is_an_error(self):
        n = _build_network()
        collapsed, mapping = collapse_mainland(n, logging.getLogger("t"))
        collapsed.set_snapshots(SNAPSHOTS[:2])
        with pytest.raises(ValueError, match="snapshots"):
            expand_results(n, collapsed, mapping, logging.getLogger("t"))
