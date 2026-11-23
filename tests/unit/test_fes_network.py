"""FES 2050 network (task Y5): boundary busmap, capability hold, branch scaling, landing table."""

import json
from pathlib import Path

import pandas as pd
import pypsa
import pytest

from scripts.network_clustering.boundary_branch_scaling import PRE_SCALING_COLUMN, scale_boundary_branches
from scripts.network_clustering.boundary_busmap import (
    boundary_sides,
    build_boundary_busmap,
    capability_boundaries,
    extend_polyline,
    straddle_check,
)
from scripts.solve.secured_boundaries import (
    DEFAULT_CAPABILITY,
    DEFAULT_GEOMETRY,
    build_secured_boundary_specs,
    capability_row,
    find_boundary_crossings,
    load_boundary_geometries,
    load_capabilities,
)

BUSMAP_FILE = Path("data/clustering/fes2050_boundary_busmap.csv")
LANDING_FILE = Path("data/network/ETYS/fes2050_offshore_landing.csv")
FLAGS_FILE = Path("data/network/ETYS/fes2050_boundary_option_flags.csv")


def _b6_network():
    """North/south of B6 (Scotland / England), a crossing line and DC link, and an internal line."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2050-01-10", periods=2, freq="h"))
    n.add("Bus", "north", x=330000.0, y=620000.0, v_nom=400.0, carrier="AC")
    n.add("Bus", "north2", x=331000.0, y=621000.0, v_nom=400.0, carrier="AC")
    n.add("Bus", "south", x=380000.0, y=520000.0, v_nom=400.0, carrier="AC")
    n.add("Bus", "south2", x=381000.0, y=519000.0, v_nom=400.0, carrier="AC")
    n.add("Line", "cross", bus0="north", bus1="south", x=10.0, r=1.0, s_nom=2000.0)
    n.add("Line", "inside", bus0="south", bus1="south2", x=10.0, s_nom=500.0)
    n.add("Line", "inside_n", bus0="north", bus1="north2", x=10.0, s_nom=500.0)
    n.add("Link", "hvdc", bus0="south2", bus1="north", p_nom=2000.0, carrier="DC", p_min_pu=-1)
    n.lines["s_nom_winter"] = n.lines["s_nom"]
    n.lines["s_nom_summer"] = n.lines["s_nom"] * 0.8
    return n


# ── boundary sides and the busmap ──────────────────────────────────────────────

def test_extended_polyline_keeps_the_original_and_goes_far():
    g = load_boundary_geometries(DEFAULT_GEOMETRY)["B6"]
    e = extend_polyline(g)
    assert e.length > g.length + 1.0e7
    assert e.distance(g.interpolate(0.5, normalized=True)) < 1e-6


def test_side_parity_matches_the_solve_crossing_test():
    n = _b6_network()
    geoms = load_boundary_geometries(DEFAULT_GEOMETRY)
    sides = boundary_sides(n.buses[["x", "y"]], geoms, ["B6"])["B6"]
    assert sides["north"] == sides["north2"] != sides["south"] == sides["south2"]
    crossing = set(find_boundary_crossings(n, geoms["B6"])["name"])
    by_parity = {name for name, r in pd.concat([n.lines[["bus0", "bus1"]], n.links[["bus0", "bus1"]]]).iterrows()
                 if sides[r.bus0] != sides[r.bus1]}
    assert crossing == by_parity == {"cross", "hvdc"}


def _one_region(tmp_path):
    """A single GSP region that B6 cuts."""
    gj = {"type": "FeatureCollection",
          "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::27700"}},
          "features": [{"type": "Feature", "properties": {"GSPs": "BIG"},
                        "geometry": {"type": "Polygon", "coordinates": [[[300000, 500000], [420000, 500000],
                                                                         [420000, 650000], [300000, 650000],
                                                                         [300000, 500000]]]}}]}
    p = tmp_path / "regions.geojson"
    p.write_text(json.dumps(gj), encoding="utf-8")
    return str(p)


def test_a_region_cut_by_a_boundary_is_split_and_merge_never_crosses_it(tmp_path):
    n = _b6_network()
    cfg = {"boundary_busmap": {"gsp_regions_path": _one_region(tmp_path), "boundaries": ["B6"]}}
    table = build_boundary_busmap(n, cfg)
    assert table["cluster"].nunique() == 2
    assert table.groupby("cluster")["side_code"].nunique().eq(1).all()
    assert table.loc["north", "cluster"].startswith("BIG__B6-")
    assert table.loc["north", "split_by"] == "B6"
    assert straddle_check(n, table["cluster"], ["B6"]).at["B6", "internal"] == 0
    cfg["boundary_busmap"]["max_clusters"] = 1          # cannot go below one cluster per side
    merged = build_boundary_busmap(n, cfg)
    assert merged["cluster"].nunique() == 2


def test_foreign_and_non_ac_buses_stay_individual(tmp_path):
    n = _b6_network()
    n.add("Bus", "FR", x=600000.0, y=100000.0, v_nom=400.0, carrier="AC", country="France")
    n.add("Bus", "H2", x=380000.0, y=520000.0, carrier="H2")
    table = build_boundary_busmap(n, {"boundary_busmap": {"gsp_regions_path": _one_region(tmp_path),
                                                           "boundaries": ["B6"]}})
    assert table.loc["FR", "cluster"] == "External_FR"
    assert table.loc["H2", "cluster"] == "H2_H2"


@pytest.mark.skipif(not BUSMAP_FILE.exists(), reason="committed busmap not present")
def test_committed_busmap_has_no_straddling_cluster_and_a_target_count():
    bm = pd.read_csv(BUSMAP_FILE, dtype={"side_code": str})
    assert bm["side_code"].str.startswith("s").all()
    assert bm.groupby("cluster")["side_code"].nunique().eq(1).all()
    assert 100 <= bm["cluster"].nunique() <= 300
    assert bm["bus_id"].is_unique
    assert {"source", "status", "base_network"} <= set(bm.columns)
    # The side code has one digit per boundary the busmap was built on (its summary's list, 30 Sep 2026).
    # B14 joined the capability file on 14 Oct (2030 item F22, ETYS 2024) after this busmap, which the
    # 2050 build no longer reads (decision 74: the NESO 17-zone busmap), so it is judged on its own list.
    built_on = json.loads(BUSMAP_FILE.with_name("fes2050_boundary_busmap_summary.json").read_text())["boundaries"]
    assert len(bm["side_code"].iloc[0]) == 1 + len(built_on)
    assert set(built_on) <= set(capability_boundaries())


# ── capability: hold the last published year ───────────────────────────────────

def test_hold_last_year_lookup():
    cap = pd.read_csv(DEFAULT_CAPABILITY)
    row, year = capability_row(cap, "B6", 2050, "HT", hold_last_year=True)
    assert year == 2045 and row["capability_mw"] == 27629
    row, year = capability_row(cap, "B6", 2050, "HT")
    assert row is None
    row, year = capability_row(cap, "B6", 2030, "HT", hold_last_year=True)
    assert year == 2030 and row["capability_mw"] == 15354       # published years are exact
    held = load_capabilities(DEFAULT_CAPABILITY, 2050, "HT", hold_last_year=True)
    at_2045 = load_capabilities(DEFAULT_CAPABILITY, 2045, "HT")
    # Each boundary is held at its own last published year: 2045 for the ETYS 2025 series, 2043 for B14
    # (ETYS 2024 trace, added 14 Oct 2026, 2030 item F22).
    pd.testing.assert_series_equal(held["capability_mw"].drop("B14"), at_2045["capability_mw"])
    row, year = capability_row(cap, "B14", 2050, "HT", hold_last_year=True)
    assert year == 2043 and held.at["B14", "capability_mw"] == row["capability_mw"] == 11600
    with pytest.raises(ValueError):
        load_capabilities(DEFAULT_CAPABILITY, 2050, "HT")


def test_specs_2050_need_the_explicit_hold_and_2030_is_unchanged():
    base = {"enabled": True, "source": "secured_capability"}
    with pytest.raises(ValueError):
        build_secured_boundary_specs(_b6_network(), {**base, "secured_capability": {"boundaries": ["B6"]}},
                                     {"modelled_year": 2050})
    (spec,) = build_secured_boundary_specs(
        _b6_network(), {**base, "secured_capability": {"boundaries": ["B6"], "hold_last_year": True}},
        {"modelled_year": 2050})
    assert spec["limit_series"].iloc[0] == pytest.approx(27629)          # January: winter
    for hold in (False, True):
        (s30,) = build_secured_boundary_specs(
            _b6_network(), {**base, "secured_capability": {"boundaries": ["B6"], "hold_last_year": hold}},
            {"modelled_year": 2030})
        assert s30["limit_series"].iloc[0] == pytest.approx(15354)


# ── branch scaling ─────────────────────────────────────────────────────────────

def _b6_crossings(n):
    return {"B6": find_boundary_crossings(n, load_boundary_geometries(DEFAULT_GEOMETRY)["B6"])}


def test_scaling_raises_crossing_branches_to_margin_times_capability():
    n = _b6_network()
    rep = scale_boundary_branches(n, _b6_crossings(n), pd.Series({"B6": 27629.0}), {"margin": 1.25})
    target_ac = 1.25 * 27629.0 - 2000.0                   # the DC link counts, and is not scaled
    assert n.lines.at["cross", "s_nom"] == pytest.approx(target_ac)
    f = target_ac / 2000.0
    assert n.lines.at["cross", "x"] == pytest.approx(10.0 / f)      # parallel circuits
    assert n.lines.at["cross", "s_nom_summer"] == pytest.approx(1600.0 * f)
    assert n.lines.at["inside", "s_nom"] == 500.0                   # internal branches untouched
    assert n.links.at["hvdc", "p_nom"] == 2000.0
    assert n.lines.at["cross", PRE_SCALING_COLUMN] == 2000.0
    assert rep.at["B6", "after_ratio"] == pytest.approx(1.25)
    again = scale_boundary_branches(n, _b6_crossings(n), pd.Series({"B6": 27629.0}), {"margin": 1.25})
    assert n.lines.at["cross", "s_nom"] == pytest.approx(target_ac)  # idempotent
    assert again.at["B6", "ac_after_mva"] == pytest.approx(target_ac)


def test_scaling_leaves_boundaries_already_above_target_and_relaxed_lines():
    n = _b6_network()
    scale_boundary_branches(n, _b6_crossings(n), pd.Series({"B6": 1000.0}), {"margin": 1.25})
    assert n.lines.at["cross", "s_nom"] == 2000.0
    n = _b6_network()
    n.lines.at["cross", "s_nom"] = 9999.0
    scale_boundary_branches(n, _b6_crossings(n), pd.Series({"B6": 27629.0}), {"margin": 1.25})
    assert n.lines.at["cross", "s_nom"] == 9999.0


def test_a_branch_on_two_boundaries_takes_the_larger_factor():
    n = _b6_network()
    cr = _b6_crossings(n)["B6"]
    rep = scale_boundary_branches(n, {"B6": cr, "X": cr.copy()}, pd.Series({"B6": 27629.0, "X": 10000.0}),
                                  {"margin": 1.0})
    assert n.lines.at["cross", "s_nom"] == pytest.approx(27629.0 - 2000.0)
    assert rep.at["X", "after_ratio"] > 1.0


def test_branch_scaling_is_off_unless_asked_and_on_through_the_specs():
    base = {"enabled": True, "source": "secured_capability"}
    n = _b6_network()
    build_secured_boundary_specs(n, {**base, "secured_capability": {"boundaries": ["B6"], "hold_last_year": True}},
                                 {"modelled_year": 2050})
    assert n.lines.at["cross", "s_nom"] == 2000.0 and PRE_SCALING_COLUMN not in n.lines
    build_secured_boundary_specs(n, {**base, "secured_capability": {
        "boundaries": ["B6"], "hold_last_year": True, "branch_scaling": {"enabled": True, "margin": 1.1}}},
        {"modelled_year": 2050})
    assert n.lines.at["cross", "s_nom"] == pytest.approx(1.1 * 27629.0 - 2000.0)


# ── committed data: landing table and option flags ──────────────────────────────

@pytest.mark.skipif(not LANDING_FILE.exists(), reason="landing table not present")
def test_landing_table_one_row_per_bus_and_fes_regional_totals():
    land = pd.read_csv(LANDING_FILE)
    assert land["landing_bus"].is_unique
    totals = land.groupby("fes_region")["capacity_mw"].sum()
    assert totals["NGET"] == pytest.approx(72335.65, abs=0.1)
    assert totals["SHETL"] == pytest.approx(23899.5, abs=0.1)
    assert totals["SPTL"] == pytest.approx(6736.0, abs=0.1)
    assert (land["v_nom"] >= 275).all()
    assert land[["source", "status", "cluster"]].notna().all().all()
    if BUSMAP_FILE.exists():
        bm = pd.read_csv(BUSMAP_FILE, usecols=["bus_id", "cluster"]).set_index("bus_id")["cluster"]
        assert set(land["landing_bus"]) <= set(bm.index)
        assert (land.set_index("landing_bus")["cluster"] == bm.reindex(land["landing_bus"]).values).all()


@pytest.mark.skipif(not FLAGS_FILE.exists(), reason="option flags not present")
def test_option_flags_cover_every_stop_and_hold_option():
    flags = pd.read_csv(FLAGS_FILE).fillna("")
    assert flags["recommendation_2026"].value_counts().to_dict() == {"Stop": 55, "Hold": 5}
    at_risk = set(flags.loc[flags["dependency_class"] == "at_risk", "code"])
    assert at_risk == {"ESC3", "JTHW", "TMPC"}
    assert flags.loc[flags["code"] == "LTRE", "superseded_by"].item() == "LTR1"
