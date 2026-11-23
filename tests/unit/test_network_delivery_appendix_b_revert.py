"""Network delivery cases undo ETYS Appendix B rows (15 Oct 2026).

The build records what every applied Appendix B row did (``ETYS_upgrades.record_appendix_b_works``,
``scripts/network_build/appendix_b_record.py``); ``market.balancing.network_delivery.revert_appendix_b``
replays each touched component without the chosen rows (``scripts/market/network_delivery.py``).

The synthetic tests run the real upgrade code on a small network whose Appendix B rows
change one circuit three times (2025, 2027, 2029), list a 2027 change before a 2025 change
on another circuit, re-terminate a circuit in 2027 and that re-termination again in 2029,
remove in 2029 a circuit a 2027 row added, and miss a 2025 change because a 2029 removal
came first. Each revert is compared with the same build without the reverted rows: the
build for an earlier year, or the build with the project's rows taken out of the workbook.

The slow tests (``slow``, ``requires_data``; set ``PYPSA_GB_RESOURCES`` to a ``resources/``
folder) build the CP30 FFR 2030 parent offline and compare its ``no_upgrades`` revert with
the 2025-only build, and its ``seven_late`` project revert with the build without those rows.
"""
import copy
import json
import logging
import os
import re
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pypsa
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.market.network_delivery import (  # noqa: E402
    _graph, apply_network_delivery, load_appendix_b_projects, network_delivery_config, project_row_id,
    select_appendix_b_rows,
)
from scripts.network_build import ETYS_upgrades as EU  # noqa: E402
from scripts.network_build.appendix_b_record import (  # noqa: E402
    APPENDIX_B_META_KEY, APPENDIX_B_ROW_ID_COLUMN, CP30_DATED_WORK_COLUMN, assign_row_ids,
    read_appendix_b_record, row_id,
)
from scripts.network_build.branch_attributes import finalize_branch_attributes  # noqa: E402
from scripts.network_build.branch_units import (  # noqa: E402
    UNITS_PU100, convert_network_pu100_to_pypsa, set_branch_units,
)

LOG = logging.getLogger(__name__)
FIELDS = ("s_nom", "r", "x", "b", "s_nom_spring", "s_nom_summer", "s_nom_autumn", "s_nom_winter")


def _row(n1, n2, year, status="Addition", r=0.01, x=0.1, b=1.0, rating=2000.0, seasons=None):
    row = {"Node1": n1, "Node2": n2, "Year": year, "Status": status, "operator": "SHE",
           "R (% on 100 MVA)": r, "X (% on 100 MVA)": x, "B (% on 100 MVA)": b,
           "Winter Rating (MVA)": rating, "OHL Length (km)": 10.0, "Cable Length (km)": 0.0,
           "Circuit Type": "OHL"}
    if seasons:
        row.update({"Spring Rating (MVA)": seasons[0], "Summer Rating (MVA)": seasons[1],
                    "Autumn Rating (MVA)": seasons[2]})
    return row


def _base():
    """Five 400 kV substations (x in m) in a ring with two chords, a 132 kV load point at AAAA."""
    n = pypsa.Network()
    set_branch_units(n, UNITS_PU100)
    for bus, x, y in (("AAAA4-", 0, 0), ("BBBB4-", 20000, 0), ("CCCC4-", 40000, 0), ("DDDD4-", 60000, 0),
                      ("EEEE4-", 30000, 20000), ("AAAA1-", 0, 0)):
        n.add("Bus", bus, v_nom=400.0 if bus[4] == "4" else 132.0, x=float(x), y=float(y), carrier="AC")
    seasons = (2130.0, 1980.0, 2130.0)
    for b0, b1, x, s_nom, seas in (("AAAA4-", "BBBB4-", 0.004, 2210.0, seasons),
                                   ("BBBB4-", "CCCC4-", 0.004, 1000.0, (np.nan,) * 3),
                                   ("AAAA4-", "CCCC4-", 0.008, 2210.0, seasons),
                                   ("CCCC4-", "DDDD4-", 0.004, 1000.0, (np.nan,) * 3),
                                   ("DDDD4-", "EEEE4-", 0.004, 800.0, seasons),
                                   ("BBBB4-", "EEEE4-", 0.004, 2210.0, seasons)):
        n.add("Line", f"{b0}_{b1}_0", bus0=b0, bus1=b1, r=x / 10, x=x, b=0.1, s_nom=s_nom, carrier="AC",
              etys_licensee="SHE", s_nom_spring=seas[0], s_nom_summer=seas[1], s_nom_autumn=seas[2])
    n.add("Transformer", "AAAA4-_AAAA1-_0", bus0="AAAA4-", bus1="AAAA1-", r=0.001, x=0.04, s_nom=240.0)
    return n


def _appendix_b():
    """Appendix B rows, in workbook order."""
    circuits = pd.DataFrame([
        # B-C: the 2027 change is listed before the 2025 one (the build applies them in this order)
        _row("BBBB4-", "CCCC4-", 2027, "Change", r=0.03, x=0.3, b=3.0, rating=2500.0),
        _row("BBBB4-", "CCCC4-", 2025, "Change", r=0.02, x=0.2, b=2.0, rating=1500.0),
        # C-D changed three times
        _row("CCCC4-", "DDDD4-", 2025, "Change", r=0.01, x=0.25, b=2.0, rating=1200.0, seasons=(1150, 1050, 1150)),
        _row("CCCC4-", "DDDD4-", 2027, "Change", r=0.02, x=0.3, b=3.0, rating=2600.0),
        _row("CCCC4-", "DDDD4-", 2029, "Change", r=0.03, x=0.35, b=4.0, rating=3100.0, seasons=(3000, 2800, 3000)),
        # A-B re-terminated into NNNN in 2027, and A-NNNN re-terminated into MMMM in 2029
        _row("AAAA4-", "BBBB4-", 2027, "Removed", rating=2210.0),
        _row("AAAA4-", "NNNN4A", 2027, rating=2780.0),
        _row("NNNN4A", "BBBB4-", 2027, rating=2780.0),
        _row("AAAA4-", "NNNN4A", 2029, "Removed", rating=2780.0),
        _row("AAAA4-", "MMMM4A", 2029, rating=2780.0),
        _row("MMMM4A", "NNNN4A", 2029, rating=2780.0),
        # B-D added in 2027 and removed in 2029
        _row("BBBB4-", "DDDD4-", 2027, rating=1800.0),
        _row("BBBB4-", "DDDD4-", 2029, "Removed", rating=1800.0),
        # D-E removed in 2029 (the build removes it before the 2025 change finds it)
        _row("DDDD4-", "EEEE4-", 2029, "Removed", rating=800.0),
        _row("DDDD4-", "EEEE4-", 2025, "Change", r=0.01, x=0.2, b=1.0, rating=900.0),
        # a 2025 addition that stays
        _row("AAAA4-", "EEEE4-", 2025, rating=1100.0),
        # generator connections: GGGG by one circuit; HHHH by two parallels (2,000 and 500 MVA)
        # or a two-circuit route through JJJJ
        _row("DDDD4-", "GGGG4-", 2027, rating=1500.0),
        _row("DDDD4-", "HHHH4-", 2027, rating=2000.0),
        _row("DDDD4-", "HHHH4-", 2027, rating=500.0),
        _row("DDDD4-", "JJJJ4-", 2027, rating=100.0),
        _row("JJJJ4-", "HHHH4-", 2027, rating=100.0),
    ])
    transformers = pd.DataFrame([
        {**_row("CCCC4-", "CCCC1-", 2027, r=0.1, x=4.0, b=0.0, rating=480.0), "Circuit Type": ""},
    ])
    return circuits, transformers


def _build(monkeypatch, year=2030, drop=(), works=None, tmp_path=None):
    """The build as ETYS_network.create_network runs it: upgrades, unit conversion, attributes."""
    circuits, transformers = _appendix_b()
    data = assign_row_ids({"circuits": circuits, "transformers": transformers, "hvdc": pd.DataFrame()})
    for key in ("circuits", "transformers"):
        data[key] = data[key][~data[key]["_row_id"].isin(set(drop))].copy()
    monkeypatch.setattr(EU, "load_etys_upgrade_data", lambda *a, **k: {k2: v.copy() for k2, v in data.items()})
    n = _base()
    n.add("Load", "load_A", bus="AAAA1-", p_set=100.0)
    audit = None if tmp_path is None else str(tmp_path / f"audit_{year}.csv")
    EU.apply_etys_network_upgrades(n, year, etys_file="x.xlsx", audit_path=audit, logger=LOG,
                                   cp30_dated_works_file=works)
    convert_network_pu100_to_pypsa(n, LOG)
    finalize_branch_attributes(n, LOG)
    return n


def _finish(n, generators=True):
    """What the later stages add: wind farms at GGGG and HHHH and a load-shedding unit on every bus."""
    if generators:
        for bus, p in (("GGGG4-", 1200.0), ("HHHH4-", 300.0)):
            if bus in n.buses.index:
                n.add("Generator", f"wind_{bus[:4]}", bus=bus, carrier="wind_onshore", p_nom=p)
    for bus in n.buses.index:
        n.add("Generator", f"{bus} load_shedding", bus=bus, carrier="load_shedding", p_nom=1e5)
    return n


def _round_trip(n, path):
    n.export_to_netcdf(path)
    return pypsa.Network(str(path))


def _revert(n, selector, projects_file=None):
    block = {"network_delivery": {"case": "test", "revert_appendix_b": selector}}
    return apply_network_delivery(n, {}, block, LOG, projects_file=projects_file)["appendix_b"]


def _assert_same_branches(n, truth, keep=()):
    for table in ("lines", "transformers"):
        a, t = getattr(n, table), getattr(truth, table)
        assert set(a.index) - set(keep) == set(t.index), table
        for name in t.index:
            assert (a.at[name, "bus0"], a.at[name, "bus1"]) == (t.at[name, "bus0"], t.at[name, "bus1"]), name
            for col in FIELDS if table == "lines" else ("s_nom", "r", "x", "b"):
                assert a.at[name, col] == pytest.approx(t.at[name, col], rel=1e-9, abs=1e-12), (name, col)


@pytest.fixture
def built(monkeypatch, tmp_path):
    return _round_trip(_finish(_build(monkeypatch, tmp_path=tmp_path)), tmp_path / "built.nc")


# ── the record ───────────────────────────────────────────────────────────────

def test_row_ids_count_identical_rows_in_workbook_order():
    df = pd.DataFrame([_row("A", "B", 2027), _row("A", "B", 2027), _row("A", "B", 2029)])
    out = assign_row_ids({"circuits": df, "transformers": pd.DataFrame(), "hvdc": pd.DataFrame()})["circuits"]
    assert list(out["_row_id"]) == ["circuit|SHE|2027|addition|A|B|1", "circuit|SHE|2027|addition|A|B|2",
                                    "circuit|SHE|2029|addition|A|B|1"]


def test_build_records_every_applied_row_in_order(built, monkeypatch, tmp_path):
    record = read_appendix_b_record(built)
    assert record["units"] == UNITS_PU100
    ids = {r["id"] for r in record["rows"]}
    assert row_id("circuit", "SHE", 2027, "removal", "AAAA4-", "BBBB4-", 1) in ids
    ops = record["ops"]
    assert [o["seq"] for o in ops] == sorted(o["seq"] for o in ops)
    cd = [o for o in ops if o["name"] == "CCCC4-_DDDD4-_0"]
    assert [o["op"] for o in cd] == ["change"] * 3
    assert [o["after"]["s_nom"] for o in cd] == [1200.0, 2600.0, 3100.0]
    assert cd[0]["before"]["s_nom"] == 1000.0
    missed = [o for o in ops if o["op"] == "change_missed"]
    assert [(o["row"], o["after"]["s_nom"]) for o in missed] == [
        (row_id("circuit", "SHE", 2025, "change", "DDDD4-", "EEEE4-", 1), 900.0)]
    audit = pd.read_csv(tmp_path / "audit_2030.csv")
    assert audit[APPENDIX_B_ROW_ID_COLUMN].notna().all()
    assert APPENDIX_B_META_KEY in built.meta


# ── selectors ────────────────────────────────────────────────────────────────

def test_after_year_selector_picks_rows_dated_after_the_year(built):
    record = read_appendix_b_record(built)
    chosen, summary = select_appendix_b_rows(record, {"after_year": 2025})
    years = {r["id"]: r["source_year"] for r in record["rows"]}
    assert chosen and all(years[r] > 2025 for r in chosen)
    assert {r for r, y in years.items() if y > 2025} == set(chosen)
    assert summary["by_year"] == {"2027": 12, "2029": 6}


def test_project_selector_reads_the_projects_file(built, tmp_path):
    projects = pd.DataFrame([{"project": "chain", "project_name": "Chained re-termination", "kind": "circuit",
                              "operation": op, "operator": "SHE", "node1": a, "node2": b, "appendix_b_year": 2029,
                              "occurrence": 1, "winter_rating_mva": 2780.0, "source": "test"}
                             for op, a, b in (("removal", "AAAA4-", "NNNN4A"), ("addition", "AAAA4-", "MMMM4A"),
                                              ("addition", "MMMM4A", "NNNN4A"))])
    path = tmp_path / "projects.csv"
    projects.to_csv(path, index=False)
    record = read_appendix_b_record(built)
    chosen, _ = select_appendix_b_rows(record, {"projects": ["chain"]}, path)
    assert set(chosen) == {project_row_id(r) for _, r in projects.iterrows()}
    with pytest.raises(ValueError, match="not in the Appendix B projects file"):
        select_appendix_b_rows(record, {"projects": ["nope"]}, path)
    projects.loc[0, "winter_rating_mva"] = 1.0
    projects.to_csv(path, index=False)
    with pytest.raises(ValueError, match="rated"):
        select_appendix_b_rows(record, {"projects": ["chain"]}, path)


@pytest.mark.parametrize("block, message", [
    ({"case": "x", "revert_cp30_dated_works": True}, "retired"),
    ({"case": "x", "revert_appendix_b": {}}, "at least one"),
    ({"case": "x", "revert_appendix_b": {"after_year": "2025"}}, "must be a year"),
    ({"case": "x", "revert_appendix_b": {"projects": "yorkshire_green"}}, "non-empty list"),
    ({"case": "x", "revert_appendix_b": {"cp30_dated_works": "yes"}}, "true or false"),
    ({"case": "x", "revert_appendix_b": {"years": 2025}}, "unknown keys"),
    ({"case": "x", "revert_appendix_b": {"after_year": 2025}, "rehome_stranded_max_mw": -1}, ">= 0"),
    ({"case": "x", "revert_appendix_b": {"after_year": 2025}, "rehome_stranded_max_mw": "50"}, ">= 0"),
])
def test_bad_revert_blocks_are_refused(block, message):
    with pytest.raises(ValueError, match=message):
        network_delivery_config({"network_delivery": block})


def test_revert_needs_the_recorded_works():
    n = pypsa.Network()
    n.add("Bus", "A")
    with pytest.raises(ValueError, match="records no Appendix B works"):
        _revert(n, {"after_year": 2025})


# ── the revert against the build without the rows ────────────────────────────

def test_after_2025_equals_the_2025_build_except_the_connection_keeps(built, monkeypatch, tmp_path):
    truth = _build(monkeypatch, year=2025)
    d = _revert(built, {"after_year": 2025})
    kept = {k["name"] for k in d["kept_as_connection"]}
    _assert_same_branches(built, truth, keep=kept)
    assert d["conflicts"] == []


def test_double_change_and_out_of_order_changes(built, monkeypatch):
    truth = _build(monkeypatch, year=2025)
    _revert(built, {"after_year": 2025})
    # C-D: three changes; only the 2025 one remains
    assert built.lines.at["CCCC4-_DDDD4-_0", "s_nom"] == 1200.0
    assert built.lines.at["CCCC4-_DDDD4-_0", "x"] == pytest.approx(truth.lines.at["CCCC4-_DDDD4-_0", "x"])
    assert built.lines.at["CCCC4-_DDDD4-_0", "s_nom_summer"] == pytest.approx(1050.0)
    # B-C: the build applied 2027 then 2025 (the 2025 values won); the 2025 build has them too
    assert built.lines.at["BBBB4-_CCCC4-_0", "s_nom"] == 1500.0 == truth.lines.at["BBBB4-_CCCC4-_0", "s_nom"]


def test_after_2027_keeps_the_2027_rows_and_undoes_2029(built, monkeypatch):
    truth = _build(monkeypatch, year=2027)
    d = _revert(built, {"after_year": 2027})
    _assert_same_branches(built, truth, keep={k["name"] for k in d["kept_as_connection"]})
    assert built.lines.at["CCCC4-_DDDD4-_0", "s_nom"] == 2600.0
    # the removal of a 2027 addition is undone: B-D is back
    assert "BBBB4-_DDDD4-_ETYS_UPG_2027" in built.lines.index
    # the chained re-termination: A-NNNN (2027) is back, the 2029 circuits through MMMM are gone
    assert "AAAA4-_NNNN4A_ETYS_UPG_2027" in built.lines.index
    assert "MMMM4A" not in built.buses.index and "AAAA4-_MMMM4A_ETYS_UPG_2029" not in built.lines.index


def test_removal_of_a_2027_addition_and_chained_re_termination_back_to_2025(built, monkeypatch):
    _revert(built, {"after_year": 2025})
    lines = built.lines
    assert "BBBB4-_DDDD4-_ETYS_UPG_2027" not in lines.index
    assert ("AAAA4-", "BBBB4-") == tuple(lines.loc["AAAA4-_BBBB4-_0", ["bus0", "bus1"]])
    assert not {"NNNN4A", "MMMM4A", "CCCC1-"} & set(built.buses.index)
    assert lines.at["AAAA4-_BBBB4-_0", "x"] == pytest.approx(0.004 * 400.0 ** 2 / 100.0)   # ohm at 400 kV


def test_a_change_the_build_missed_is_replayed_on_the_restored_circuit(built, monkeypatch):
    truth = _build(monkeypatch, year=2025)
    d = _revert(built, {"after_year": 2025})
    restored = [r for r in d["restored"] if r["name"] == "DDDD4-_EEEE4-_0"]
    assert restored and restored[0]["replayed_rows"] == [
        row_id("circuit", "SHE", 2025, "change", "DDDD4-", "EEEE4-", 1)]
    assert built.lines.at["DDDD4-_EEEE4-_0", "s_nom"] == 900.0 == truth.lines.at["DDDD4-_EEEE4-_0", "s_nom"]


def test_project_selector_equals_the_build_without_the_project_rows(built, monkeypatch, tmp_path):
    rows = [row_id("circuit", "SHE", 2029, op, a, b, 1) for op, a, b in (
        ("removal", "AAAA4-", "NNNN4A"), ("addition", "AAAA4-", "MMMM4A"), ("addition", "MMMM4A", "NNNN4A"))]
    truth = _build(monkeypatch, drop=rows)
    d = _revert(built, {"rows": rows})
    assert d["rows_selected"] == 3 and d["kept_as_connection"] == [] and d["conflicts"] == []
    _assert_same_branches(built, truth)
    assert "AAAA4-_NNNN4A_ETYS_UPG_2027" in built.lines.index and "MMMM4A" not in built.buses.index


# ── the keep rule ────────────────────────────────────────────────────────────

def test_keep_rule_fewest_branches_lowest_rating_and_report(built):
    d = _revert(built, {"after_year": 2025})
    kept = {k["name"]: k for k in d["kept_as_connection"]}
    # GGGG: its one circuit; HHHH: the 500 MVA parallel, not the 2,000 MVA one or the two-circuit route
    assert set(kept) == {"DDDD4-_GGGG4-_ETYS_UPG_2027", "DDDD4-_HHHH4-_ETYS_UPG_2027_1"}
    g = kept["DDDD4-_GGGG4-_ETYS_UPG_2027"]
    assert g["connects"] == ["GGGG4-"] and g["s_nom"] == 1500.0 and g["occupant_mw"] == 1200.0
    assert g["rows"] == [row_id("circuit", "SHE", 2027, "addition", "DDDD4-", "GGGG4-", 1)]
    assert kept["DDDD4-_HHHH4-_ETYS_UPG_2027_1"]["s_nom"] == 500.0
    assert "DDDD4-_HHHH4-_ETYS_UPG_2027" not in built.lines.index and "JJJJ4-" not in built.buses.index
    assert nx.number_connected_components(_graph(built)) == 1
    # every kept branch is radial: removing it would cut its connection off
    bridges = {frozenset(e) for e in nx.bridges(nx.Graph(_graph(built)))}
    assert all(frozenset((k["bus0"], k["bus1"])) in bridges for k in kept.values())


def test_a_hydrogen_bus_is_not_a_route_to_the_grid(monkeypatch, tmp_path):
    """Electrolysers share one GB_H2 bus; through it GGGG looked connected and lost its circuit (23 Oct 2026)."""
    n = _finish(_build(monkeypatch))
    n.add("Bus", "GB_H2", carrier="H2")
    for bus in ("GGGG4-", "AAAA4-"):
        n.add("Link", f"electrolysis_{bus}", bus0=bus, bus1="GB_H2", p_nom=500.0, carrier="electrolysis")
    n = _round_trip(n, tmp_path / "h2.nc")
    d = _revert(n, {"after_year": 2025})
    assert "DDDD4-_GGGG4-_ETYS_UPG_2027" in {k["name"] for k in d["kept_as_connection"]}
    assert nx.number_connected_components(_graph(n).subgraph(n.buses.index[n.buses.carrier == "AC"])) == 1


def test_without_occupants_nothing_is_kept(monkeypatch, tmp_path):
    n = _round_trip(_finish(_build(monkeypatch), generators=False), tmp_path / "bare.nc")
    truth = _build(monkeypatch, year=2025)
    d = _revert(n, {"after_year": 2025})
    assert d["kept_as_connection"] == []
    _assert_same_branches(n, truth)
    assert set(n.buses.index) == set(truth.buses.index)


# ── re-homing small embedded plant on stranded new buses (F25.5, 27 Oct 2026) ──

def _small_plant(monkeypatch, tmp_path, at_gggg=30.0, at_hhhh=300.0, load=False, rehome=None):
    n = _build(monkeypatch)
    n.add("Generator", "solar_GGGG", bus="GGGG4-", carrier="solar_pv", p_nom=at_gggg * 0.6)
    n.add("StorageUnit", "battery_GGGG", bus="GGGG4-", carrier="Battery", p_nom=at_gggg * 0.4, max_hours=2.0)
    n.add("Generator", "wind_HHHH", bus="HHHH4-", carrier="wind_onshore", p_nom=at_hhhh)
    if load:
        n.add("Load", "load_GGGG", bus="GGGG4-", p_set=5.0)
    n = _round_trip(_finish(n, generators=False), tmp_path / "small.nc")
    block = {"network_delivery": {"case": "test", "revert_appendix_b": {"after_year": 2025}}}
    if rehome is not None:
        block["network_delivery"]["rehome_stranded_max_mw"] = rehome
    return n, apply_network_delivery(n, {}, block, LOG)["appendix_b"]


def test_small_embedded_plant_on_a_stranded_new_bus_moves_to_the_nearest_bus(monkeypatch, tmp_path):
    n, d = _small_plant(monkeypatch, tmp_path)
    kept = {k["name"] for k in d["kept_as_connection"]}
    assert kept == {"DDDD4-_HHHH4-_ETYS_UPG_2027_1"}        # HHHH's 300 MW keeps its 500 MVA parallel
    moved = {m["name"]: m for m in d["rehomed"]}
    assert set(moved) == {"solar_GGGG", "battery_GGGG"}
    assert {m["to_bus"] for m in moved.values()} == {"DDDD4-"}                  # nearest 400 kV bus, 4.9 km
    assert all(m["same_voltage"] and m["from_bus"] == "GGGG4-" for m in moved.values())
    assert moved["solar_GGGG"]["distance_km"] == pytest.approx(4.9, abs=0.1)
    assert n.generators.at["solar_GGGG", "bus"] == "DDDD4-" and n.storage_units.at["battery_GGGG", "bus"] == "DDDD4-"
    assert n.generators.at["solar_GGGG", "p_nom"] == pytest.approx(18.0)          # capacity unchanged
    assert "GGGG4-" not in n.buses.index and "DDDD4-_GGGG4-_ETYS_UPG_2027" not in n.lines.index
    assert d["stranded_pockets_kept"] == [{"buses": ["HHHH4-"], "occupant_mw": 300.0,
                                           "reason": "above the threshold"}]
    assert d["rehome_max_mw"] == 50.0
    assert json.loads(n.meta["network_delivery"])["appendix_b"]["rehomed"] == json.loads(json.dumps(d["rehomed"]))


def test_a_stranded_new_bus_with_load_or_above_the_threshold_keeps_its_circuit(monkeypatch, tmp_path):
    _, with_load = _small_plant(monkeypatch, tmp_path, load=True)
    assert with_load["rehomed"] == []
    assert "DDDD4-_GGGG4-_ETYS_UPG_2027" in {k["name"] for k in with_load["kept_as_connection"]}
    assert {"buses": ["GGGG4-"], "occupant_mw": 30.0, "reason": "load"} in with_load["stranded_pockets_kept"]
    _, big = _small_plant(monkeypatch, tmp_path, at_gggg=60.0)
    assert big["rehomed"] == [] and "DDDD4-_GGGG4-_ETYS_UPG_2027" in {k["name"] for k in big["kept_as_connection"]}
    _, raised = _small_plant(monkeypatch, tmp_path, at_gggg=60.0, rehome=100.0)
    assert {m["name"] for m in raised["rehomed"]} == {"solar_GGGG", "battery_GGGG"}


def test_rehoming_off_keeps_every_connection_as_before(monkeypatch, tmp_path):
    n, d = _small_plant(monkeypatch, tmp_path, rehome=0)
    assert d["rehomed"] == [] and d["stranded_pockets_kept"] == []
    assert {k["name"] for k in d["kept_as_connection"]} == {"DDDD4-_GGGG4-_ETYS_UPG_2027",
                                                             "DDDD4-_HHHH4-_ETYS_UPG_2027_1"}
    assert n.generators.at["solar_GGGG", "bus"] == "GGGG4-"


def test_record_round_trips_and_the_delivery_record_is_saved(built, tmp_path):
    d = _revert(built, {"after_year": 2025})
    m = _round_trip(built, tmp_path / "delivered.nc")
    assert json.loads(m.meta["network_delivery"])["appendix_b"] == json.loads(json.dumps(d))
    assert APPENDIX_B_META_KEY in m.meta


def test_cp30_dated_works_selector_and_tags(monkeypatch, tmp_path):
    works = pd.DataFrame([{"project_code": "ADD", "in_service_year": 2030, "component": "circuit",
                           "operation": "addition", "etys_sheet": "B-2-2a", "operator": "SHE",
                           "node1": "CCCC4-", "node2": "EEEE4-", "appendix_b_year": 2032, "occurrence": 1,
                           "r": 0.01, "x": 0.1, "b": 1.0, "impedance_units": "percent_on_100MVA",
                           "winter_rating_mva": 2780.0, "source": "test source"}])
    works.to_csv(tmp_path / "works.csv", index=False)
    circuits, transformers = _appendix_b()
    circuits = pd.concat([circuits, pd.DataFrame([_row("CCCC4-", "EEEE4-", 2032, rating=2780.0)])],
                         ignore_index=True)
    monkeypatch.setattr(EU, "load_etys_upgrade_data", lambda *a, **k: {
        "circuits": circuits.copy(), "transformers": transformers.copy(), "hvdc": pd.DataFrame()})
    n = _base()
    n.add("Load", "load_A", bus="AAAA1-", p_set=100.0)
    EU.apply_etys_network_upgrades(n, 2030, etys_file="x.xlsx", logger=LOG,
                                   cp30_dated_works_file=str(tmp_path / "works.csv"))
    convert_network_pu100_to_pypsa(n, LOG)
    finalize_branch_attributes(n, LOG)
    assert n.lines.at["CCCC4-_EEEE4-_ETYS_UPG_2032", CP30_DATED_WORK_COLUMN] == "ADD"
    _finish(n)
    d = _revert(n, {"cp30_dated_works": True})
    assert d["rows_selected"] == 1 and "CCCC4-_EEEE4-_ETYS_UPG_2032" not in n.lines.index
    assert "AAAA4-_MMMM4A_ETYS_UPG_2029" in n.lines.index           # everything else stays


# ── config and data ──────────────────────────────────────────────────────────

def test_projects_file_names_the_two_seven_late_projects():
    table = load_appendix_b_projects()
    assert set(table["project"]) == {"yorkshire_green", "brinsworth_chesterfield_high_marnham"}
    assert set(table["operator"]) == {"NGET"}
    assert set(table.loc[table.project == "yorkshire_green", "appendix_b_year"]) == {2027}
    assert set(table.loc[table.project != "yorkshire_green", "appendix_b_year"]) == {2029}
    ends = set(table["node1"].str[:4]) | set(table["node2"].str[:4])
    assert {"OSBA", "OVER", "POPP", "BRIN", "CHTE", "HIMB"} <= ends


def test_delivery_cases_in_the_scenarios():
    """no_upgrades reverts every row after 2025; seven_late reverts its two projects; no_acceleration neither."""
    sys.path.insert(0, str(ROOT / "config"))
    from config_loader import load_config
    seen = {}
    months_by_tag = {}
    for sid, cfg in load_config()["scenarios"].items():
        block = ((cfg.get("market") or {}).get("balancing") or {}).get("network_delivery")
        if not block:
            continue
        network_delivery_config({"network_delivery": block})
        case = block["case"]
        family = "no_upgrades" if case.startswith("no_upgrades") else "seven" if case.startswith("seven") else case
        seen[family] = seen.get(family, 0) + 1
        dated = re.match(r"CP30_FFR_2030_([A-Z][a-z]{2})_(\w+)$", sid)
        if dated:
            months_by_tag.setdefault((dated.group(2), family), set()).add(dated.group(1))
        selector = block.get("revert_appendix_b")
        if family == "no_upgrades":
            assert selector == {"after_year": 2025}, sid
            assert "Appendix B row dated after 2025" in block["description"], sid
            assert not any("HVDC" in (e.get("name") or "") for e in block.get("remove_components", [])), sid
        elif family == "seven":
            assert selector == {"projects": ["yorkshire_green", "brinsworth_chesterfield_high_marnham"]}, sid
            assert "reverted in the topology" in block["description"], sid
        else:
            assert selector is None, sid
    # 19 Oct 2026: the count of delivery-case months grows with every family the delay study defines
    # (the network-bound bid stacks and central-dispatch years added 12 at a time), so the test
    # counts what is defined rather than pinning a total: every reverting family is a whole year,
    # and the solved families the bill's network bounds read are present.
    assert seen.get("no_upgrades", 0) >= 12 and seen.get("seven", 0) >= 12
    whole_year = {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"}
    for (tag, family), months in months_by_tag.items():
        if family in ("no_upgrades", "seven"):
            assert months == whole_year, (tag, sorted(whole_year - months))
    for tag, family in (("no_upgrades", "no_upgrades"), ("seven_late", "seven")):
        assert months_by_tag.get((tag, family)) == whole_year, tag
    assert any(tag == "no_acceleration" for tag, _ in months_by_tag), "the CENTRAL network's market months"


# ── the CP30 FFR parent network, built offline ───────────────────────────────

FFR = "CP30_FFR_2030"
NO_UPGRADES = "CP30_FFR_2030_Jan_no_upgrades"
SEVEN_LATE = "CP30_FFR_2030_Jan_seven_late"


def _resources():
    root = Path(os.environ.get("PYPSA_GB_RESOURCES", ROOT / "resources"))
    needed = [root / "network/ETYS_2024_components.csv", root / "network/ETYS_2024_buses.csv",
              root / "CP30/CP30_reference_N1_N2_NA1.csv"]
    return root if all(p.exists() for p in needed) else None


def _ffr_build(resources, out, upgrade_year=None, cp30_on=True, drop=()):
    sys.path.insert(0, str(ROOT / "config"))
    from config_loader import load_config
    from scripts.network_build import ETYS_network as EN
    cfg = load_config()["scenarios"][FFR]
    df = pd.read_csv(resources / "network/ETYS_2024_components.csv", index_col=0)
    buses = pd.read_csv(resources / "network/ETYS_2024_buses.csv", index_col=0)
    offshore = set(buses[buses["is_offshore"] == True].index)  # noqa: E712
    buses = EN.guess_GSP_location_of_remaining_buses(df, buses.drop(columns=["is_offshore"]), offshore, LOG)
    buses, offshore = EN.apply_bus_coordinate_overrides(
        buses, offshore, str(ROOT / "data/network/ETYS/bus_coordinate_overrides.csv"), LOG)
    eu = cfg["network"]["upgrades"]
    cp30 = dict(cfg["network"]["reinforcements"], reference_file=str(resources / "CP30/CP30_reference_N1_N2_NA1.csv")) \
        if cp30_on else {"enabled": False}
    original = EU.load_etys_upgrade_data
    if drop:
        def without(*a, **k):
            data = assign_row_ids(original(*a, **k))
            for key in ("circuits", "transformers"):
                data[key] = data[key][~data[key]["_row_id"].isin(set(drop))].copy()
            return data
        EU.load_etys_upgrade_data = without
    try:
        EN.create_network(
            df, buses, LOG, export_path=str(out / "network.nc"),
            etys_upgrade_audit_path=str(out / "audit.csv"), cp30_reinforcement_config=cp30,
            cp30_reinforcement_audit_path=str(out / "cp30_audit.csv"),
            hnd_hybrid_config=cfg.get("network", {}).get("hnd_hybrid", {}), hnd_hybrid_audit_path=str(out / "hnd.csv"),
            etys_upgrades_enabled=True,
            upgrade_year=int(upgrade_year or eu.get("upgrade_year") or cfg["modelled_year"]),
            etys_upgrade_file=str(ROOT / "data/network/ETYS/ETYS 2024 Appendix-B V1.xlsx"),
            substation_coords_file=str(ROOT / "data/network/ETYS/substation_coordinates.csv"),
            topology_sensitivities=eu.get("topology_sensitivities", []),
            documented_topology_corrections=eu.get("documented_topology_corrections", {}),
            engineering_fallbacks=eu.get("engineering_fallbacks", {}),
        )
    finally:
        EU.load_etys_upgrade_data = original
    return pypsa.Network(str(out / "network.nc"))


@pytest.fixture(scope="module")
def ffr(tmp_path_factory):
    resources = _resources()
    if resources is None:
        pytest.skip("processed ETYS tables not available (set PYPSA_GB_RESOURCES to a resources/ folder)")
    sys.path.insert(0, str(ROOT / "config"))
    from config_loader import load_config
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        built = _ffr_build(resources, tmp_path_factory.mktemp("ffr2030"))
        truth = _ffr_build(resources, tmp_path_factory.mktemp("ffr2025"), upgrade_year=2025, cp30_on=False)
        drop = [project_row_id(r) for _, r in load_appendix_b_projects().iterrows()]
        without = _ffr_build(resources, tmp_path_factory.mktemp("ffr_without"), drop=drop)
        scenarios = load_config()["scenarios"]
    finally:
        os.chdir(cwd)
    return built, truth, without, scenarios[NO_UPGRADES], scenarios[SEVEN_LATE]


def _slack(n):
    for bus in n.buses.index:
        n.add("Generator", f"{bus} load_shedding", bus=bus, carrier="load_shedding", p_nom=1e5)
    return n


@pytest.mark.slow
@pytest.mark.requires_data
def test_ffr_no_upgrades_equals_the_2025_build(ffr):
    built, truth, _, scenario, _ = ffr
    n = _slack(built.copy())
    record = apply_network_delivery(n, copy.deepcopy(scenario), copy.deepcopy(scenario["market"]["balancing"]), LOG)
    d = record["appendix_b"]
    assert d["rows_selected"] > 1000 and d["conflicts"] == [] and d["kept_as_connection"] == []
    _assert_same_branches(n, truth)
    live = set(n.links.index[n.links.p_nom > 0])
    assert live == set(truth.links.index[truth.links.p_nom > 0])
    for link in ("BRNX4A_HAWP41_HVDC_2029", "DRAX41_PEHE4J_HVDC_2029", "BANN4R_LOSD4R_HVDC_2030"):
        assert n.links.at[link, "p_nom"] == 0.0
    # extra buses: the ends of the disabled links and the CP30 layer's own, all isolated
    extra = set(n.buses.index) - set(truth.buses.index)
    assert extra <= set(d.get("buses_kept_for_disabled_links", [])) | {"EACS41", "TILN41"}
    n.consistency_check()


@pytest.mark.slow
@pytest.mark.requires_data
def test_ffr_seven_late_projects_equal_the_build_without_their_rows(ffr):
    built, _, without, _, scenario = ffr
    n = _slack(built.copy())
    selector = scenario["market"]["balancing"]["network_delivery"]["revert_appendix_b"]
    d = _revert(n, selector)
    assert d["rows_selected"] == len(load_appendix_b_projects()) and d["conflicts"] == []
    _assert_same_branches(n, without)
    assert set(n.buses.index) == set(without.buses.index)
