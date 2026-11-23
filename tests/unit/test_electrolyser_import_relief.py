"""Demand-side headroom (spatial_utils.NodeImportModel) and the electrolyser import relief (16 Oct 2026).

The calibration solve of 15 Oct 2026 shed 215 GWh at BUMU3-, ABNE3- and BAIN3-: electrolysers held at
their Stage 1 schedule on 132 kV busbars that could not import them. The model judges each node's
demand (its own and its substation's lower-voltage busbars) on the circuits that demand must cross,
each loaded by its DC share; the relief moves electrolysis off any node above max_loading x that.
"""
import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.hydrogen.add_hydrogen_system import (
    place_fes_electrolysers,
    relieve_electrolyser_import_headroom,
    resolve_electrolyser_siting,
)
from scripts.utilities.spatial_utils import NodeImportModel, branch_x_pu, node_import_headroom
from config.config_loader import load_layer

LOG = logging.getLogger("test")
NO_FLOOR = {"min_bm_constraint_voltage_kv": 0.0}


# ----------------------------------------------------------------------------- NodeImportModel
def _model_inputs(buses, lines=(), transformers=()):
    """buses: (name, v_nom); lines/transformers: (name, bus0, bus1, rating, x_pu)."""
    b = pd.DataFrame({"v_nom": [v for _, v in buses]}, index=[n for n, _ in buses])
    rows = [dict(name=n, bus0=a, bus1=c, rating=r, x_pu=x, kind="line") for n, a, c, r, x in lines]
    rows += [dict(name=n, bus0=a, bus1=c, rating=r, x_pu=x, kind="transformer") for n, a, c, r, x in transformers]
    branches = pd.DataFrame(rows).set_index("name")
    line_table = pd.DataFrame([dict(name=n, bus0=a, bus1=c, s_nom=r) for n, a, c, r, _ in lines]).set_index("name") \
        if lines else pd.DataFrame(columns=["bus0", "bus1", "s_nom"])
    return SimpleNamespace(buses=b, lines=line_table), branches


def _bumu(coupled=True, x_left=1e-3, x_right=1e-3):
    """GRID4- source; SITE1Q/1R 132 kV busbars (coupled or not) each with a 132 MVA circuit, to LEFT1- and RITE1-;
    SITE3- a 33 kV load busbar under both."""
    buses = [("GRID4-", 400.0), ("LEFT1-", 132.0), ("RITE1-", 132.0), ("SITE1Q", 132.0), ("SITE1R", 132.0),
             ("SITE3-", 33.0)]
    lines = [("g_l", "GRID4-", "LEFT1-", 5000.0, 1e-4), ("g_r", "GRID4-", "RITE1-", 5000.0, 1e-4),
             ("c_q", "LEFT1-", "SITE1Q", 132.0, x_left), ("c_r", "RITE1-", "SITE1R", 132.0, x_right)]
    if coupled:
        lines.append(("coupler", "SITE1Q", "SITE1R", 9999.0, 1e-6))
    transformers = [("t_q", "SITE1Q", "SITE3-", 90.0, 3e-3), ("t_r", "SITE1R", "SITE3-", 90.0, 3e-3)]
    return _model_inputs(buses, lines, transformers)


def test_load_busbar_draws_through_its_132kv_node_and_its_transformers_are_not_capacity():
    network, branches = _bumu()
    out = node_import_headroom(network, branches, pd.Series({"SITE1Q": 390.0, "SITE3-": 50.0}))
    row = out.loc["SITE1Q"]
    assert row["buses"] == "SITE1Q|SITE1R"
    assert row["demand_mw"] == pytest.approx(440.0)
    assert row["import_mw"] == pytest.approx(264.0)          # the two 132 kV circuits, not the 2 x 90 MVA transformers
    assert row["capacity_mw"] == pytest.approx(264.0, rel=1e-3)  # equal impedance: equal shares (coupler x ~ 0)
    assert row["ratio"] == pytest.approx(440.0 / 264.0, rel=1e-3)
    assert out.loc["SITE3-", "ratio"] < 1.0                  # the load busbar itself has 180 MVA of transformers


def test_uncoupled_busbars_do_not_share_their_circuits():
    network, branches = _bumu(coupled=False)
    out = node_import_headroom(network, branches, pd.Series({"SITE1Q": 100.0, "SITE3-": 145.0}))
    q = out.loc["SITE1Q"]
    # Summing the site's circuits would give 245 / 264 = 0.93; SITE1Q's own circuit carries its own 100 MW
    # (most of it) and half of SITE3-'s 145 MW.
    assert q["binding"] == "c_q"
    assert q["ratio"] > 1.0
    assert 0.5 < q["binding_share"] < 1.0
    assert out.loc["SITE1R", "ratio"] < q["ratio"]


def test_parallel_circuits_share_by_impedance_not_rating():
    network, branches = _bumu(x_left=1e-3, x_right=3e-3)
    out = node_import_headroom(network, branches, pd.Series({"SITE1Q": 200.0}))
    row = out.loc["SITE1Q"]
    assert row["binding"] == "c_q"                        # the lower-impedance circuit takes more
    assert row["binding_share"] == pytest.approx(0.75, abs=0.02)        # about 3:1, the upstream lines shift it
    assert row["capacity_mw"] == pytest.approx(132.0 / row["binding_share"], rel=1e-6)
    assert row["capacity_mw"] < 264.0


def test_relaxed_circuits_never_bind_and_a_dead_end_busbar_is_judged_on_its_site():
    network, branches = _bumu()
    branches.loc[["c_q", "c_r"], "rating"] = np.inf
    out = node_import_headroom(network, branches, pd.Series({"SITE1Q": 1000.0}))
    assert out.loc["SITE1Q", "ratio"] == 0.0
    # A 400 kV busbar fed only through its own substation's transformer (BLYT41 under BLYT21): its pocket is the
    # site, its circuits the site's, so it has capacity (it is not a node with nothing feeding it).
    network, branches = _model_inputs(
        [("GRID4-", 400.0), ("BLYT41", 400.0), ("BLYT21", 275.0)],
        lines=[("ext", "GRID4-", "BLYT21", 1000.0, 1e-3)],
        transformers=[("sgt", "BLYT41", "BLYT21", 1148.0, 1e-4)])
    out = node_import_headroom(network, branches, pd.Series({"BLYT21": 200.0}))
    assert out.loc["BLYT41", "capacity_mw"] == pytest.approx(1000.0, rel=1e-6)
    assert out.loc["BLYT41", "ratio"] == pytest.approx(0.2, rel=1e-6)


def test_branch_x_pu_follows_pypsa_bases():
    buses = pd.DataFrame({"v_nom": [132.0, 33.0]}, index=["A", "B"])
    lines = pd.DataFrame({"bus0": ["A"], "bus1": ["A"], "x": [17.424]}, index=["l"])
    transformers = pd.DataFrame({"bus0": ["A"], "bus1": ["B"], "x": [0.2], "s_nom": [60.0]}, index=["t"])
    x = branch_x_pu(buses, lines, transformers)
    assert x["l"] == pytest.approx(17.424 / 132.0 ** 2)
    assert x["t"] == pytest.approx(0.2 / 60.0)


# ----------------------------------------------------------------------------- relief
def _relief_network(extra_400=()):
    """GRID4- (400 kV) feeds OTHR1- (132 kV), which feeds ABNE1Q by one 100 MVA circuit; ABNE3- (33 kV) hangs off ABNE1Q.
    ``extra_400``: (name, x_km, y_km) 400 kV buses each tied to GRID4- by a 2,000 MVA circuit."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-15", periods=2, freq="h"))
    for name, v, x, y in [("GRID4-", 400.0, 0, 0), ("OTHR1-", 132.0, 5000, 0), ("ABNE1Q", 132.0, 10000, 0),
                          ("ABNE3-", 33.0, 10000, 0)] + [(b, 400.0, x * 1000, y * 1000) for b, x, y in extra_400]:
        n.add("Bus", name, v_nom=v, x=x, y=y, carrier="AC")
    n.add("Transformer", "sgt_oth", bus0="GRID4-", bus1="OTHR1-", s_nom=2000.0, x=0.1, r=0.0)
    n.add("Line", "oth_abne", bus0="OTHR1-", bus1="ABNE1Q", s_nom=100.0, x=10.0, r=0.1)
    n.add("Transformer", "abne_33", bus0="ABNE1Q", bus1="ABNE3-", s_nom=1000.0, x=0.2, r=0.0)
    for b, _, _ in extra_400:
        n.add("Line", f"grid_{b}", bus0="GRID4-", bus1=b, s_nom=2000.0, x=5.0, r=0.05)
    n.add("Load", "L_ABNE", bus="ABNE3-", p_set=pd.Series([20.0, 10.0], index=n.snapshots))
    return n


def _rows(**mw_by_bus):
    return pd.DataFrame([{"site_name": f"electrolysis_FES_{b}", "bus": b, "power_mw": mw, "area": "SHETL",
                          "gsp": b, "fes_row": "gsp", "technology": "electrolysis"} for b, mw in mw_by_bus.items()])


def _geography(n, region=None, north=None):
    region = region or {}
    north = north or {}
    bus_region = pd.Series({b: region.get(b, "North Scotland") for b in n.buses.index})
    sides = pd.DataFrame({k: [north.get(b, True) if k == "B4" else True for b in n.buses.index]
                          for k in ("B4", "B6", "B8")}, index=n.buses.index)
    return bus_region, sides


SITING = resolve_electrolyser_siting({"electrolyser_siting": {"method": "fes_gsp"}})


def test_relief_moves_the_excess_to_the_nearest_400kv_node_in_region_and_side():
    # CLOS4- is nearer but in another region, SIDE4- nearer still but across B4: NEAR4- takes it.
    n = _relief_network([("CLOS4-", 12, 0), ("SIDE4-", 11, 0), ("NEAR4-", 30, 0)])
    bus_region, sides = _geography(n, region={"CLOS4-": "South Scotland", "GRID4-": "South Scotland"},
                                   north={"SIDE4-": False})
    rows, moves = relieve_electrolyser_import_headroom(_rows(**{"ABNE3-": 300.0}), n, SITING, NO_FLOOR,
                                                       bus_region, sides, LOG)
    assert rows["power_mw"].sum() == pytest.approx(300.0, rel=1e-12)
    by_bus = rows.groupby("bus")["power_mw"].sum()
    # ABNE keeps what its 100 MVA circuit carries with its 20 MW peak load.
    assert by_bus["ABNE3-"] + 20.0 == pytest.approx(100.0, abs=0.05)
    assert by_bus["ABNE3-"] + 20.0 <= 100.0
    assert set(moves["to_bus"]) == {"NEAR4-"} and moves["method"].iloc[0] == "nearest_20km"
    assert "electrolysis_FES_ABNE3-_relief_NEAR4-" in set(rows["site_name"])   # deterministic name


def test_relief_prefers_a_higher_voltage_node_of_the_same_substation():
    n = _relief_network([("NEAR4-", 15, 0)])
    n.add("Bus", "ABNE4-", v_nom=400.0, x=10000, y=0, carrier="AC")
    n.add("Line", "grid_abne4", bus0="GRID4-", bus1="ABNE4-", s_nom=2000.0, x=5.0, r=0.05)
    n.add("Transformer", "abne_sgt", bus0="ABNE4-", bus1="ABNE1Q", s_nom=60.0, x=0.2, r=0.0)
    bus_region, sides = _geography(n)
    rows, moves = relieve_electrolyser_import_headroom(_rows(**{"ABNE1Q": 300.0}), n, SITING, NO_FLOOR,
                                                       bus_region, sides, LOG)
    first = moves.iloc[0]
    assert first["to_bus"] == "ABNE4-" and first["method"] == "same_substation"
    # A withdrawal at ABNE4- returns partly through OTHR1- and the SGT: a sliver goes to the next 400 kV node.
    assert moves.groupby("to_bus")["capacity_mw"].sum()["ABNE4-"] > 0.95 * moves["capacity_mw"].sum()
    assert rows["power_mw"].sum() == pytest.approx(300.0, rel=1e-12)


def test_relief_leaves_a_node_within_its_limit_alone_and_reports_one_it_cannot_fix():
    n = _relief_network([("NEAR4-", 30, 0)])
    bus_region, sides = _geography(n)
    rows, moves = relieve_electrolyser_import_headroom(_rows(**{"ABNE3-": 60.0}), n, SITING, NO_FLOOR,
                                                       bus_region, sides, LOG)
    assert moves.empty and rows["bus"].tolist() == ["ABNE3-"]
    # No 400 kV node in the region: nothing moves.
    bus_region, sides = _geography(n, region={"NEAR4-": "South Scotland", "GRID4-": "South Scotland"})
    rows, moves = relieve_electrolyser_import_headroom(_rows(**{"ABNE3-": 300.0}), n, SITING, NO_FLOOR,
                                                       bus_region, sides, LOG)
    assert moves.empty and rows["power_mw"].sum() == pytest.approx(300.0)


def test_siting_needs_the_relief_geography_and_retires_the_search_radius():
    assert SITING["max_loading"] == 1.0 and SITING["min_target_voltage_kv"] == 275.0
    assert SITING["boundaries"] == ["B4", "B6", "B8"]
    with pytest.raises(ValueError, match="retired"):
        resolve_electrolyser_siting({"electrolyser_siting": {"method": "fes_gsp", "search_radius_km": 40.0}})
    n = _relief_network()
    fes = pd.DataFrame([{"Building Block ID Number": "Dem_BB009", "FES Pathway": "Holistic Transition",
                         "GSP": "Abernethy", "Node ID": "ABNE3-", "DNO License Area": "SSEN (SHEPD)", "2030": 50.0}])
    with pytest.raises(ValueError, match="relief geography"):
        place_fes_electrolysers(n, fes, "Holistic Transition", 2030, 50.0, SITING, LOG)
    off = dict(SITING, max_loading=0.0)
    assert place_fes_electrolysers(n, fes, "Holistic Transition", 2030, 50.0, off, LOG).to_dict() == {"ABNE3-": 50.0}


def test_defaults_yaml_electrolyser_siting_matches_the_relief_settings():
    import yaml
    from pathlib import Path

    from config.config_loader import deep_merge

    # the siting machinery (method overlay) with CP30's FES edition (source overlay), as a CP30 scenario merges them
    layers = deep_merge(load_layer("method", "brownfield"), load_layer("source", "CP30"))
    siting = layers["hydrogen"]["electrolyser_siting"]
    assert (siting["fes_year"], siting["fes_scenario"]) == (SITING["fes_year"], SITING["fes_scenario"])
    resolved = resolve_electrolyser_siting({"electrolyser_siting": siting})
    assert resolved["method"] == "fes_gsp"
    for key in ("min_target_voltage_kv", "boundaries", "fes_year", "fes_scenario"):
        assert resolved[key] == SITING[key], key
    # CP30 keeps a 10% margin for through-flow at ABNE (user, 16 Oct 2026); the code default stays 1.0.
    assert resolved["max_loading"] == 0.9 and SITING["max_loading"] == 1.0
