"""The CP30 fleet location check (rule validate_CP30_fleet_location, 19 Oct 2026)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from shapely.geometry import LineString

from scripts.CP30.fleet_location_validation import (
    DEFAULT_CONFIG,
    boundary_sides,
    bus_positions,
    circuit_positions,
    etys_line_lengths,
    fleet_location_config,
    interconnector_links,
    position_exemptions,
    position_rows,
    read_es5,
    run_checks,
    scenario_settings,
    status_of,
)
from config.config_loader import load_layer

ROOT = Path(__file__).resolve().parents[2]
KM = 1e3


# ── Config ──────────────────────────────────────────────────────────────────
@pytest.mark.unit
def test_code_defaults_and_unknown_keys():
    assert fleet_location_config(None) == DEFAULT_CONFIG
    assert fleet_location_config({"min_mw": 10.0})["fail_mw"] == DEFAULT_CONFIG["fail_mw"]
    with pytest.raises(ValueError):
        fleet_location_config({"not_a_key": 1})


@pytest.mark.unit
def test_defaults_block_equals_the_code_defaults():
    """validation.fleet_location of the CP30 source overlay must equal DEFAULT_CONFIG."""
    block = (load_layer("source", "CP30").get("validation") or {}).get("fleet_location")
    assert block is not None, "validation.fleet_location must be in config/defaults/source/CP30.yaml"
    assert block == DEFAULT_CONFIG
    assert fleet_location_config(block) == DEFAULT_CONFIG


@pytest.mark.unit
def test_scenario_settings_pass_only_what_the_check_reads():
    sc = {"CP30_pathway": "Further Flex and Renewables", "modelled_year": 2030, "big": {"x": 1},
          "validation": {"fleet_location": {"min_mw": 10.0}},
          "network": {"transmission": {"outage_schedule": {"secured_capability": {
              "include": {"B4": [{"component": "link", "name": "X", "sign": -1.0}]}, "capability_year": 2030}}}}}
    s = scenario_settings(sc)
    assert s == {"cp30_fleet_location_validation": {"min_mw": 10.0},
                 "include": {"B4": [{"component": "link", "name": "X", "sign": -1.0}]}, "geometry_file": None,
                 "pathway": "Further Flex and Renewables", "modelled_year": 2030}


@pytest.mark.unit
def test_status_thresholds():
    cfg = fleet_location_config(None)
    assert status_of(100.0, np.nan, cfg) == "INFO"
    assert status_of(1000.0, 1020.0, cfg) == "PASS"        # 20 MW: below min_mw
    assert status_of(1000.0, 1100.0, cfg) == "WARN"        # 100 MW, 9%
    assert status_of(1000.0, 1600.0, cfg) == "FAIL"        # beyond fail_mw
    assert status_of(100.0, 300.0, cfg) == "FAIL"          # 67% and above fail_mw / 4
    assert status_of(30.0, 0.0, cfg) == "WARN"


# ── A small synthetic network ───────────────────────────────────────────────
#: Two boundaries drawn west to east, so their left (+1) is north: B4 at y = 500 km, B6 at y = 300 km.
GEOMS = {"B4": LineString([(0, 500 * KM), (1000 * KM, 500 * KM)]),
         "B6": LineString([(0, 300 * KM), (1000 * KM, 300 * KM)])}
BUS_XY = {"N1": (100, 600), "N2": (110, 610), "M1": (100, 400), "S1": (100, 200), "S2": (120, 200),
          "P1": (900, 200),                                     # misplaced: its circuits say it is near S1/S2
          "A1": (300, 200), "A2": (305, 205), "B1": (350, 200), "B2": (355, 195), "B3": (360, 210),
          "S1X": (100, 203), "FR_External": (900, 50)}


def tiny_net() -> dict:
    buses = pd.DataFrame({"x": [v[0] * KM for v in BUS_XY.values()], "y": [v[1] * KM for v in BUS_XY.values()],
                          "lon": np.nan, "lat": np.nan, "v_nom": 400.0, "carrier": "AC",
                          "country": ["GB"] * (len(BUS_XY) - 1) + ["FR"]}, index=list(BUS_XY))
    lines = pd.DataFrame([
        ("N1_N2", "N1", "N2", 15.0), ("N1_M1", "N1", "M1", 200.0), ("M1_S1", "M1", "S1", 200.0),
        ("S1_S2", "S1", "S2", 20.0), ("S1_P1", "S1", "P1", 10.0), ("S2_P1", "S2", "P1", 10.0),
        # A short ETYS section length on a longer route (Eccles-Stella West): one circuit, neither bus suspect.
        ("A1_B1", "A1", "B1", 5.0), ("A1_A2", "A1", "A2", 8.0), ("A1_S2", "A1", "S2", 180.0),
        ("B1_B2", "B1", "B2", 8.0), ("B1_B3", "B1", "B3", 15.0),
    ], columns=["name", "bus0", "bus1", "length_km"]).set_index("name").assign(s_nom=1000.0, length=0.0)
    trafos = pd.DataFrame({"bus0": ["S1"], "bus1": ["S1X"], "s_nom": [500.0]}, index=["T_S1"])
    gens = pd.DataFrame([
        ("on_n", "N1", "wind_onshore", 900.0, "", "", ""),
        ("on_m", "M1", "wind_onshore", 500.0, "", "", ""),
        ("on_s", "S1", "wind_onshore", 300.0, "", "", ""),
        ("farm_ok", "N2", "wind_offshore", 600.0, "Farm OK", "1", "named"),
        ("farm_far", "S1", "wind_offshore", 400.0, "Farm Far", "2", "named"),
        ("proxy", "S2", "wind_offshore", 250.0, "", "", "proxy"),
    ], columns=["name", "bus", "carrier", "p_nom", "site", "repd_ref_id", "fleet_origin"]).set_index("name")
    gens = gens.assign(lon=-2.0, lat=55.0)
    sus = pd.DataFrame({"bus": ["N1", "S2"], "carrier": ["Battery", "pumped_hydro"], "p_nom": [100.0, 200.0]},
                       index=["batt", "psh"])
    links = pd.DataFrame({"bus0": ["S2", "FR_External", "N1"], "bus1": ["FR_External", "S2", "S1"],
                          "carrier": ["DC", "DC", "DC"], "p_nom": [1000.0, 1000.0, 2000.0],
                          "interconnector_name": ["IC1", "IC1", ""]}, index=["IC_IC1", "IC_IC1_rev", "HVDC_NS"])
    return dict(buses=buses, gens=gens, sus=sus, links=links, lines=lines, trafos=trafos)


@pytest.mark.unit
def test_boundary_sides_are_topological_and_take_the_include():
    net = tiny_net()
    sides = boundary_sides(net, GEOMS, ["B4", "B6", "B9"], include=None)
    assert "B9" not in sides                        # no geometry: skipped
    # The internal HVDC link N1-S1 crosses both polylines and is cut like the AC circuits.
    assert sides.at["N1", "B4"] == 1 and sides.at["M1", "B4"] == -1 and sides.at["S1", "B6"] == -1
    assert sides.at["M1", "B6"] == 1 and sides.at["P1", "B6"] == -1 and sides.at["FR_External", "B4"] == 0
    # A declared include cuts a branch the geometry misses: N1-N2 on B4 puts N2 on the other side.
    sides2 = boundary_sides(net, GEOMS, ["B4"], include={"B4": [{"component": "line", "name": "N1_N2", "sign": 1.0}]})
    assert sides2.at["N1", "B4"] == 1 and sides2.at["N2", "B4"] == -1


@pytest.mark.unit
def test_interconnectors_counted_once_with_their_gb_bus():
    ic = interconnector_links(tiny_net())
    assert list(ic["name"]) == ["IC1"] and list(ic["bus"]) == ["S2"]


@pytest.mark.unit
def test_position_check_flags_the_misplaced_bus_only():
    cfg = fleet_location_config(None)
    net = tiny_net()
    circuits = circuit_positions(net, None, cfg)
    bad = set(circuits.index[circuits["inconsistent"]])
    assert bad == {"S1_P1", "S2_P1", "A1_B1", "T_S1"}
    per_bus = bus_positions(circuits, cfg)
    # P1: both its circuits over by far more than 10 km: misplaced (FAIL).
    assert per_bus.at["P1", "verdict"] == "misplaced"
    # The short section: one inconsistent circuit of three at each end.
    assert per_bus.at["A1", "verdict"] == "unresolved" and per_bus.at["B1", "verdict"] == "unresolved"
    # S1: two of four (P1 and the transformer, ends 3 km apart): not most; S1X has one circuit only.
    assert per_bus.at["S1", "verdict"] == "unresolved" and per_bus.at["S1X", "verdict"] == "unresolved"
    rows, _ = position_rows(net, None, cfg)
    table = pd.DataFrame(rows)
    assert list(table.loc[table["status"].eq("FAIL"), "key"]) == ["P1"]
    # Every inconsistent circuit WARNs; buses with a minority of inconsistent circuits are INFO.
    circ = table[table["scope"].str.startswith("circuit")]
    assert set(circ["key"]) == bad and set(circ["status"]) == {"WARN"}
    assert set(table.loc[table["scope"].eq("bus") & table["key"].ne("P1"), "status"]) == {"INFO"}


@pytest.mark.unit
def test_position_lengths_prefer_the_etys_table(tmp_path):
    cfg = fleet_location_config(None)
    comp = tmp_path / "ETYS_2024_components.csv"
    pd.DataFrame({"name": ["S1_P1", "S2_P1", "T_S1"], "component": ["line", "line", "transformer"],
                  "length_km": [800.0, 780.0, 0.0]}).to_csv(comp, index=False)
    lengths = etys_line_lengths(comp)
    assert list(lengths.index) == ["S1_P1", "S2_P1"]
    circuits = circuit_positions(tiny_net(), lengths, cfg)
    assert circuits.at["S1_P1", "length_source"] == "etys_components" and not circuits.at["S1_P1", "inconsistent"]
    assert circuits.at["N1_N2", "length_source"] == "network"
    assert bus_positions(circuits, cfg).at["P1", "verdict"] == "ok"
    assert etys_line_lengths(tmp_path / "missing.csv").empty


@pytest.mark.unit
def test_inter_site_lines_without_a_length_are_not_judged():
    cfg = fleet_location_config(None)
    net = tiny_net()
    net["lines"].loc["S1_P1", "length_km"] = 0.0          # a wind-farm edge with no length
    circuits = circuit_positions(net, None, cfg)
    assert "S1_P1" not in circuits.index
    net["lines"].loc["N1_N2", "length_km"] = np.nan       # N1 and N2 are different sites
    assert "N1_N2" not in circuit_positions(net, None, cfg).index


@pytest.mark.unit
def test_read_es5_from_the_extracted_csv(tmp_path):
    path = tmp_path / "es5.csv"
    pd.DataFrame({"source_sheet": "ES5", "Scenario": ["FFR", "FFR", "Other"], "Technology": ["Solar", "Solar", "Solar"],
                  "Variable": ["Capacity (MW)", "Generation (GWh)", "Capacity (MW)"], "Wales": [10.0, 99.0, 5.0]}
                 ).to_csv(path, index=False)
    es5 = read_es5(path, "FFR")
    assert es5.to_dict() == {"Wales": {"Solar": 10.0}}


def _es5() -> pd.DataFrame:
    return pd.DataFrame({"North Scotland": [900.0, 600.0, 0.0, 100.0, 100.0],
                         "South Scotland": [500.0, 0.0, 0.0, 0.0, 0.0],
                         "Midlands": [300.0, 650.0, 0.0, 0.0, 200.0]},
                        index=["Onshore Wind", "Offshore Wind", "Solar", "Batteries", "Storage"])


@pytest.mark.unit
def test_run_checks_end_to_end(tmp_path):
    cfg = fleet_location_config(None)
    net = tiny_net()
    gb = [b for b in BUS_XY if b != "FR_External"]
    zone = {"N1": "Z1", "N2": "Z1", "M1": "Z4"}
    bus_zone = pd.DataFrame({"neso_zone": [zone.get(b, "Z9") for b in gb], "mapping": "polygon",
                             "reconciled": False}, index=gb)
    region = {"N1": "North Scotland", "N2": "North Scotland", "M1": "South Scotland"}
    bus_region = pd.Series({b: region.get(b, "Midlands") for b in gb})
    landing = tmp_path / "landing.csv"
    pd.DataFrame({"repd_ref_id": ["1", "2"], "landing_bus": ["N2", "N1"]}).to_csv(landing, index=False)
    icref = pd.DataFrame({"composition_name": ["IC1"], "accepted_buses": ["S2"], "source": ["register"]})
    comp = pd.DataFrame({"name": ["IC1"], "cp30_mw": ["1000"], "first_year": [""], "action": ["keep"]})
    table, summary, buses = run_checks(
        net, cfg=cfg, es5=_es5(), geoms=GEOMS, include=None, gsp_polygons=None, gsp_group_region=None,
        landing_files=[landing], ic_reference=icref, ic_composition=comp, model_year=2030,
        bus_zone=bus_zone, bus_region=bus_region)
    t = table.set_index(["check", "key", "technology"])
    assert t.at[("es5_region", "North Scotland", "Onshore wind"), "status"] == "PASS"
    assert t.at[("es5_region", "Midlands", "Offshore wind"), "ours_mw"] == pytest.approx(650.0)
    assert t.at[("boundary_remote_side", "B6", "Onshore wind"), "cp30_mw"] == pytest.approx(1400.0)
    assert t.at[("boundary_remote_side", "B6", "Onshore wind"), "status"] == "PASS"
    assert t.at[("offshore_landing", "Farm OK", "Offshore wind"), "status"] == "PASS"
    far = t.loc[("offshore_landing", "Farm Far", "Offshore wind")]
    assert far["status"] == "FAIL" and "B4" in far["note"] and "B6" in far["note"]
    assert t.at[("offshore_landing", "S2", "Offshore wind"), "status"] == "INFO"     # proxy bus of 250 MW
    assert t.at[("interconnector", "IC1", "Interconnector"), "status"] == "PASS"
    assert t.at[("interconnector", "GB", "Interconnector"), "cp30_mw"] == pytest.approx(1000.0)
    assert t.at[("position", "P1", "position"), "status"] == "FAIL"
    assert summary["position_misplaced_buses"] == ["P1"] and summary["position_fail_buses"] == ["P1"]
    assert summary["status_counts"]["FAIL"] == 2                        # the Farm Far landing and P1
    json.dumps(summary, default=str)
    # A wrong interconnector landing FAILs.
    icref.loc[0, "accepted_buses"] = "S1"
    table2, _, _ = run_checks(
        net, cfg=cfg, es5=_es5(), geoms=GEOMS, include=None, gsp_polygons=None, gsp_group_region=None,
        landing_files=[landing], ic_reference=icref, ic_composition=comp, model_year=2030,
        bus_zone=bus_zone, bus_region=bus_region)
    assert table2.set_index(["check", "key", "technology"]).at[("interconnector", "IC1", "Interconnector"), "status"] == "FAIL"


# ── Position FAIL rules and exemptions (bus position audit, 19 Oct 2026) ──
@pytest.mark.unit
def test_position_thresholds_are_1_1_length_plus_3_km_and_2_km_for_transformers():
    cfg = fleet_location_config(None)
    net = tiny_net()
    net["lines"].loc["N1_N2", "length_km"] = 9.0           # 14.1 km apart: allowed 1.1 x 9 + 3 = 12.9
    c = circuit_positions(net, None, cfg)
    assert c.at["N1_N2", "allowed_km"] == pytest.approx(12.9) and c.at["N1_N2", "inconsistent"]
    assert c.at["T_S1", "allowed_km"] == pytest.approx(2.0) and c.at["T_S1", "inconsistent"]


@pytest.mark.unit
def test_exempt_circuits_are_info_and_do_not_count(tmp_path):
    cfg = fleet_location_config(None)
    path = tmp_path / "position_exemptions.csv"
    pd.DataFrame({"bus0": ["B1"], "bus1": ["A1"], "reason": ["ETYS section length on a longer route"],
                  "source": ["test"]}).to_csv(path, index=False)
    ex = position_exemptions(path)
    rows, per_bus = position_rows(tiny_net(), None, cfg, exemptions=ex)
    table = pd.DataFrame(rows).set_index("key")
    assert table.at["A1_B1", "status"] == "INFO" and "exempt" in table.at["A1_B1", "note"]
    assert per_bus.at["A1", "n_inconsistent"] == 0           # the exempt circuit is not counted
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"bus0": ["B1"], "bus1": ["A1"], "reason": [""]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="reason"):
        position_exemptions(bad)
    assert position_exemptions(tmp_path / "missing.csv").empty


@pytest.mark.unit
def test_a_far_circuit_across_a_boundary_fails():
    cfg = fleet_location_config(None)
    net = tiny_net()
    # M1 (y = 400 km) to S1 (y = 200 km) crosses B6 at y = 300 km; give it a 150 km length (allowed 168).
    net["lines"].loc["M1_S1", "length_km"] = 150.0
    rows, _ = position_rows(net, None, cfg, geoms=GEOMS, boundaries=["B4", "B6"])
    table = pd.DataFrame(rows).set_index("key")
    assert table.at["M1_S1", "status"] == "FAIL" and "crosses B6" in table.at["M1_S1", "note"]
    # Over by less than boundary_excess_km: WARN only.
    net["lines"].loc["M1_S1", "length_km"] = 177.0           # allowed 197.7: 200 km apart, over by 2.3 km
    rows, _ = position_rows(net, None, cfg, geoms=GEOMS, boundaries=["B4", "B6"])
    assert pd.DataFrame(rows).set_index("key").at["M1_S1", "status"] == "WARN"


@pytest.mark.unit
def test_a_misplaced_end_failing_on_region_zone_or_boundary_side():
    cfg = fleet_location_config(None)
    net = tiny_net()
    # Q1 sits north of B4 (y = 600 km) but its two circuits (12 km to S1 and S2) put it near S1/S2 (y = 200 km).
    net["buses"].loc["Q1"] = {"x": 110 * KM, "y": 600 * KM, "lon": np.nan, "lat": np.nan, "v_nom": 400.0,
                              "carrier": "AC", "country": "GB"}
    for name, other in (("S1_Q1", "S1"), ("S2_Q1", "S2")):
        net["lines"].loc[name] = {"bus0": other, "bus1": "Q1", "length_km": 12.0, "s_nom": 1000.0, "length": 0.0}
    region = lambda x, y: np.where(np.asarray(y) > 500 * KM, "North Scotland", "Midlands")  # noqa: E731
    zone = lambda x, y: np.where(np.asarray(y) > 500 * KM, "Z1", "Z9")  # noqa: E731
    rows, per_bus = position_rows(net, None, cfg, geoms=GEOMS, boundaries=["B4", "B6"],
                                  bus_mw=pd.Series({"Q1": 400.0}), point_region=region, point_zone=zone)
    table = pd.DataFrame(rows)
    q = table.set_index("key").loc["Q1"]
    assert q["status"] == "FAIL"
    assert "region Midlands" in q["note"] and "zone Z9" in q["note"] and "across B4,B6" in q["note"]
    assert q["moved_km"] > 300
    # The well-placed ends (S1, S2) are not moved: the circuits blame Q1.
    assert table.query("scope == 'bus' and status == 'FAIL'")["key"].tolist() == ["P1", "Q1"]


@pytest.mark.unit
def test_far_transformer_fails_and_low_voltage_array_strings_are_not_judged():
    cfg = fleet_location_config(None)
    net = tiny_net()
    net["buses"].loc["S1X", "x"] = 115 * KM                    # 15 km from S1: transformer ends too far apart
    rows, _ = position_rows(net, None, cfg)
    assert pd.DataFrame(rows).set_index("key").at["T_S1", "status"] == "FAIL"
    # A 33 kV same-site line without a length is an offshore array string, not a bus section.
    for name, x in (("WFRMA3", 500), ("WFRMB3", 520)):
        net["buses"].loc[name] = {"x": x * KM, "y": 500 * KM, "lon": np.nan, "lat": np.nan, "v_nom": 33.0,
                                  "carrier": "AC", "country": "GB"}
    net["lines"].loc["WFRMA3_WFRMB3"] = {"bus0": "WFRMA3", "bus1": "WFRMB3", "length_km": np.nan, "s_nom": 500.0,
                                         "length": 0.0}
    assert "WFRMA3_WFRMB3" not in circuit_positions(net, None, cfg).index


@pytest.mark.unit
def test_committed_position_exemptions_are_reviewed():
    ex = position_exemptions(ROOT / DEFAULT_CONFIG["position_exemptions_file"])
    pairs = {frozenset(p) for p in zip(ex["bus0"], ex["bus1"])}
    assert frozenset(("ECCL4C", "STWB4Q")) in pairs and frozenset(("LOYW12", "LOAW34")) in pairs
    assert (ex["reason"].str.len() > 20).all() and (ex["source"].str.len() > 0).all()
