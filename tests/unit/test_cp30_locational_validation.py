"""Tests for the CP30 locational and network validation (scripts/CP30/locational_validation.py).

A synthetic three-region world: North Scotland above y = 500 km, South Scotland between
300 and 500 km, East England below 300 km; boundary B4 at y = 500 km and B6 at y = 300 km.
Every check has a PASS and a FAIL (or WARN) case. The live-network tests run only where the
built network exists (CP30_LOCVAL_NETWORK, or the default resources path).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

gpd = pytest.importorskip("geopandas")
from shapely.geometry import LineString, box  # noqa: E402

from scripts.CP30 import locational_validation as lv  # noqa: E402

REGIONS = ["North Scotland", "South Scotland", "East England"]
PATHWAY = "Further Flex and Renewables"
KM = 1000.0


# ── synthetic world ────────────────────────────────────────────────────────────

def _geoms():
    return {"B4": LineString([(-100 * KM, 500 * KM), (1100 * KM, 500 * KM)]),
            "B6": LineString([(-100 * KM, 300 * KM), (1100 * KM, 300 * KM)])}


def _polygons():
    return gpd.GeoDataFrame(
        {"GSPGroup": ["_P", "_N", "_A"], "GSPs": ["AAAA_1|BBBB_1", "CCCC_1|EEEE_1", "DDDD_1"]},
        geometry=[box(0, 500 * KM, 1000 * KM, 1000 * KM), box(0, 300 * KM, 1000 * KM, 500 * KM),
                  box(0, 0, 1000 * KM, 300 * KM)], crs="EPSG:27700")


def _group_region():
    return pd.Series({"_P": "North Scotland", "_N": "South Scotland", "_A": "East England"})


def _lonlat(x, y):
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    return t.transform(x, y)


BUSES = {  # name: (x km, y km, v_nom)
    "AAAA41": (200, 800, 400), "BBBB41": (600, 700, 400), "CCCC41": (300, 400, 400), "EEEE41": (700, 450, 400),
    "DDDD41": (300, 100, 400), "DDDD11": (320, 120, 132), "FFFF11": (500, 150, 132), "POCK1-": (210, 820, 132),
}


def _config(**over):
    cfg = {
        "boundary_remote_points": {"default": [500 * KM, 950 * KM]},
        "offshore": {"proxy_boundaries": ["B4", "B6"], "crossing_exceptions": []},
        "electrolysis": {"boundaries": ["B4", "B6"]},
        "ldes": {"boundaries": ["B4", "B6"]},
        "demand": {"boundaries": ["B4", "B6"]},
        "exposure": {"boundaries": ["B4", "B6"]},
        "es5_regional": {"declared_overshoots": []},
        "named_stations": {"exceptions": {}},
    }
    return lv.resolve_config(_merge(cfg, over))


def _merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _tables(generators=None, storage=None, links=None, loads=None, load_p=None, lines=None):
    buses = pd.DataFrame({"x": [v[0] * KM for v in BUSES.values()], "y": [v[1] * KM for v in BUSES.values()],
                          "v_nom": [v[2] for v in BUSES.values()], "carrier": "AC", "country": "GB"},
                         index=list(BUSES))
    gen_cols = ["bus", "p_nom", "carrier", "lat", "lon", "fleet_origin", "cp30_region", "repd_ref_id",
                "source_site_name"]
    generators = generators if generators is not None else pd.DataFrame(columns=gen_cols)
    storage = storage if storage is not None else pd.DataFrame(columns=["bus", "p_nom", "carrier", "lat", "lon"])
    links = links if links is not None else pd.DataFrame(columns=["bus0", "bus1", "p_nom", "carrier",
                                                                  "interconnector_name"])
    if loads is None:
        loads = pd.DataFrame({"bus": ["AAAA41", "CCCC41", "DDDD41"], "carrier": "electricity"},
                             index=["L_A", "L_C", "L_D"])
        load_p = pd.DataFrame({"L_A": [10.0, 20.0], "L_C": [30.0, 40.0], "L_D": [60.0, 140.0]})
    if lines is None:
        lines = pd.DataFrame({"bus0": ["AAAA41", "BBBB41", "CCCC41", "EEEE41", "DDDD41", "AAAA41"],
                              "bus1": ["BBBB41", "CCCC41", "DDDD41", "CCCC41", "EEEE41", "POCK1-"],
                              "s_nom": [5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 100.0]},
                             index=[f"L{i}" for i in range(6)])
    transformers = pd.DataFrame({"bus0": ["DDDD41", "DDDD41"], "bus1": ["DDDD11", "FFFF11"], "s_nom": [500.0, 500.0]},
                                index=["T0", "T1"])
    return lv.NetworkTables(buses, generators, storage, links, loads, lines, transformers, load_p,
                            pd.Series(np.ones(len(load_p))))


def _es5(**values):
    """ES5 table: values[tech] = (NS, SS, EE)."""
    defaults = {"Onshore Wind": (0, 0, 0), "Offshore Wind": (0, 0, 0), "Solar": (0, 0, 0), "Batteries": (0, 0, 0),
                "Storage": (0, 0, 0)}
    defaults.update(values)
    rows = [dict(source_sheet="ES5", Scenario=PATHWAY, Technology=t, Variable="Capacity (MW)", **dict(zip(REGIONS, v)))
            for t, v in defaults.items()]
    return pd.DataFrame(rows)


def _dem_per_node(rows=()):
    """GB_network.xlsx Dem_per_node rows (GSP Id, Node Id, share %) for the GSP-node locator of check 8."""
    return pd.DataFrame(list(rows), columns=["GSP Id", "Node Id", "Dem as % of GSP"])


def _run(tables, check, refs=None, es5=None, ed1=None, es1=None, site_regions=None, transmission=None,
         dem_per_node=None, **cfg):
    return lv.run_locational_checks(
        tables, _config(**cfg), es5=es5 if es5 is not None else _es5(), ed1=ed1, es1=es1, pathway=PATHWAY,
        year=2030, refs=refs or {}, geoms=_geoms(), polygons=_polygons(), group_region=_group_region(),
        checks=[check], site_regions=site_regions or {},
        sources={"gsp_dem_per_node": dem_per_node if dem_per_node is not None else _dem_per_node()},
        transmission=transmission or {"min_bm_constraint_voltage_kv": 220.0,
                                      "bm_voltage_floor": {"scottish_transmission_constrained": True,
                                                           "scottish_min_constrained_kv": 132.0}})


def _gen(rows):
    df = pd.DataFrame(rows).set_index("name")
    for col in ["fleet_origin", "cp30_region", "repd_ref_id", "source_site_name", "lat", "lon"]:
        if col not in df:
            df[col] = np.nan
    return df


def _status(table, check, item_contains):
    sel = table[(table["check"] == check) & table["item"].str.contains(item_contains, regex=False)]
    assert len(sel), f"no row {check} / {item_contains}: {table[['check', 'item', 'status']].to_string()}"
    return list(sel["status"])


# ── geography ──────────────────────────────────────────────────────────────────

def test_bus_regions_default_to_the_shared_transmission_site_table():
    from scripts.CP30.spatial_allocation import TRANSMISSION_SITE_REGIONS
    buses = pd.DataFrame({"x": [200 * KM, 200 * KM], "y": [800 * KM, 800 * KM], "carrier": "AC", "country": "GB"},
                         index=["CRUA2Q", "ZZZZ2Q"])
    region = lv.assign_bus_regions(buses, _polygons(), _group_region())
    assert region["ZZZZ2Q"] == "North Scotland" and region["CRUA2Q"] == TRANSMISSION_SITE_REGIONS["CRUA"]


def test_regions_and_sides_follow_polygons_and_overrides():
    t = _tables()
    region = lv.assign_bus_regions(t.buses, _polygons(), _group_region(), {"EEEE": "North Scotland"})
    assert region["AAAA41"] == "North Scotland" and region["CCCC41"] == "South Scotland"
    assert region["DDDD41"] == "East England" and region["EEEE41"] == "North Scotland"  # override wins
    sides = lv.oriented_sides(t.buses, _geoms(), ["B4", "B6"], {"default": [500 * KM, 950 * KM]})
    assert sides.loc["AAAA41", "B4"] and not sides.loc["CCCC41", "B4"] and sides.loc["CCCC41", "B6"]
    assert not sides.loc["DDDD41", "B6"]


def test_resolve_config_deep_merges_over_defaults():
    cfg = lv.resolve_config({"demand": {"share_warn_pp": 1.0}})
    assert cfg["demand"]["share_warn_pp"] == 1.0
    assert cfg["demand"]["share_fail_pp"] == lv.DEFAULT_CONFIG["demand"]["share_fail_pp"]


def test_defaults_yaml_block_equals_code_defaults():
    """validation.locational of config/defaults/source/CP30.yaml must equal DEFAULT_CONFIG."""
    yaml = pytest.importorskip("yaml")
    from config.config_loader import load_defaults, load_layer

    defaults = load_defaults()
    assert "locational" not in (defaults.get("validation") or {})   # a CP30 block: not merged into other scenarios
    assert load_layer("source", "CP30")["validation"]["locational"] == lv.DEFAULT_CONFIG


# ── check 1: ES5 regional ──────────────────────────────────────────────────────

def test_es5_regional_pass_by_bus_region():
    g = _gen([{"name": "w1", "bus": "AAAA41", "p_nom": 1000.0, "carrier": "wind_onshore", "cp30_region": "North Scotland"},
              {"name": "w2", "bus": "CCCC41", "p_nom": 500.0, "carrier": "wind_onshore", "cp30_region": "South Scotland"}])
    out = _run(_tables(generators=g), "es5_regional", es5=_es5(**{"Onshore Wind": (1000, 500, 0)}))
    rows = out[out["item"].str.startswith("wind_onshore |") & ~out["item"].str.contains("stored")]
    assert set(rows["status"]) == {"PASS"}


def test_es5_regional_fails_when_bus_is_in_another_region_despite_tag():
    # Tagged South Scotland (as ES5 counts it) but connected north of B4: the bus decides.
    g = _gen([{"name": "w1", "bus": "AAAA41", "p_nom": 1500.0, "carrier": "wind_onshore", "cp30_region": "North Scotland"},
              {"name": "w2", "bus": "BBBB41", "p_nom": 500.0, "carrier": "wind_onshore", "cp30_region": "South Scotland"}])
    out = _run(_tables(generators=g), "es5_regional", es5=_es5(**{"Onshore Wind": (1500, 500, 0)}))
    assert _status(out, "es5_regional", "wind_onshore | North Scotland") == ["FAIL"]
    assert _status(out, "es5_regional", "wind_onshore | South Scotland") == ["FAIL"]
    assert _status(out, "es5_regional", "wind_onshore | stored cp30_region differs") == ["WARN"]


def test_es5_regional_declared_overshoot_is_carried_nationally():
    g = _gen([{"name": "w1", "bus": "AAAA41", "p_nom": 1100.0, "carrier": "wind_onshore"},
              {"name": "w2", "bus": "CCCC41", "p_nom": 400.0, "carrier": "wind_onshore"}])
    es5 = _es5(**{"Onshore Wind": (1000, 500, 0)})
    over = {"declared_overshoots": [{"technology": "wind_onshore", "region": "North Scotland", "max_mw": 120.0}]}
    out = _run(_tables(generators=g), "es5_regional", es5=es5, es5_regional=over)
    assert _status(out, "es5_regional", "wind_onshore | North Scotland") == ["PASS"]
    assert _status(out, "es5_regional", "wind_onshore | South Scotland") == ["PASS"]
    out = _run(_tables(generators=g), "es5_regional", es5=es5)  # undeclared: both fail
    assert _status(out, "es5_regional", "wind_onshore | North Scotland") == ["FAIL"]


def test_non_battery_storage_region_uses_override(tmp_path):
    s = pd.DataFrame({"bus": ["EEEE41"], "p_nom": [440.0], "carrier": ["LAES"]}, index=["CP30_LAES_existing_X"])
    es5 = _es5(**{"Storage": (0, 440, 0)})
    out = _run(_tables(storage=s), "es5_regional", es5=es5)
    assert _status(out, "es5_regional", "non_battery_storage | South Scotland") == ["PASS"]
    out = _run(_tables(storage=s), "es5_regional", es5=es5, site_regions={"EEEE": "North Scotland"})
    assert _status(out, "es5_regional", "non_battery_storage | North Scotland") == ["FAIL"]


# ── check 2: offshore ──────────────────────────────────────────────────────────

def _landing_ref():
    return pd.DataFrame({"repd_ref_id": [1, 2, 3], "site_name": ["Farm A", "Farm B", "Gone C"],
                         "landing_bus": ["AAAA41", "DDDD41", "CCCC41"], "evidence": ["tec_register", "none", "tec"],
                         "source": ["TEC", "proxy", "TEC"], "excluded_not_built": [False, False, True],
                         "sourced": [True, False, True]})


def test_offshore_named_pass_and_fail():
    lon, lat = _lonlat(200 * KM, 800 * KM)
    ok = _gen([{"name": "A", "bus": "AAAA41", "p_nom": 500.0, "carrier": "wind_offshore", "fleet_origin": "named",
                "repd_ref_id": 1, "source_site_name": "Farm A", "cp30_region": "North Scotland", "lat": lat, "lon": lon}])
    out = _run(_tables(generators=ok), "offshore", refs={"offshore_landing_reference": _landing_ref()})
    assert "FAIL" not in set(out[out["check"] == "offshore_named"]["status"])
    bad = ok.assign(bus="BBBB41")
    out = _run(_tables(generators=bad), "offshore", refs={"offshore_landing_reference": _landing_ref()})
    assert _status(out, "offshore_named", "Farm A") == ["FAIL"]
    unsourced = bad.assign(repd_ref_id=2, source_site_name="Farm B")
    out = _run(_tables(generators=unsourced), "offshore", refs={"offshore_landing_reference": _landing_ref()})
    assert _status(out, "offshore_named", "Farm B") == ["WARN"]


def test_offshore_refill_must_land_at_an_excluded_projects_landing():
    g = _gen([{"name": "CP30_increment_wind_offshore_refill_South_Scotland_CCCC41_2030", "bus": "CCCC41",
               "p_nom": 100.0, "carrier": "wind_offshore", "fleet_origin": "increment", "cp30_region": "South Scotland"}])
    out = _run(_tables(generators=g), "offshore", refs={"offshore_landing_reference": _landing_ref()})
    assert _status(out, "offshore_refill", "refill at CCCC41") == ["PASS"]
    out = _run(_tables(generators=g.assign(bus="EEEE41")), "offshore",
               refs={"offshore_landing_reference": _landing_ref()})
    assert _status(out, "offshore_refill", "refill at EEEE41") == ["FAIL"]


def test_offshore_proxy_crossing_boundary_fails_and_exception_reports_info():
    # The Blackhillock case: a South Scotland (Fife) proxy landed north of B4.
    lon, lat = _lonlat(300 * KM, 420 * KM)
    g = _gen([{"name": "CP30_increment_wind_offshore_0_AR7", "bus": "AAAA41", "p_nom": 2000.0,
               "carrier": "wind_offshore", "fleet_origin": "increment", "cp30_region": "South Scotland",
               "lat": lat, "lon": lon}])
    out = _run(_tables(generators=g), "offshore")
    assert _status(out, "offshore_proxy", "B4: proxy landing at AAAA41") == ["FAIL"]
    assert _status(out, "offshore_proxy", "B6: proxies landing across") == ["PASS"]
    assert _status(out, "offshore_region", "tag South Scotland | landing AAAA41") == ["FAIL"]
    exc = {"crossing_exceptions": [{"boundary": "B4", "landing_bus": "AAAA41", "reason": "test"}],
           "region_exceptions": [{"landing_bus": "AAAA41", "reason": "test"}]}
    out = _run(_tables(generators=g), "offshore", offshore=exc)
    assert _status(out, "offshore_proxy", "B4: proxy landing at AAAA41") == ["INFO"]
    good = g.assign(bus="CCCC41")
    out = _run(_tables(generators=good), "offshore")
    assert set(out[out["check"] == "offshore_proxy"]["status"]) == {"PASS"}
    assert _status(out, "offshore_region", "tag equals") == ["PASS"]


# ── check 3: named stations ────────────────────────────────────────────────────

def _station_ref():
    return pd.DataFrame({"station": ["Big CCGT", "Nuke", "Small"], "component": "generator",
                         "carrier": ["CCGT", "nuclear", "CCGT"], "match": ["Big CCGT", "Nuke.*", "Small"],
                         "expected_buses": ["DDDD41", "CCCC41", "DDDD11"], "required": ["false", "true", "false"],
                         "tec_connection_site": "X", "source": "TEC"})


def test_named_station_pass_fail_exception_and_missing():
    g = _gen([{"name": "Big CCGT", "bus": "DDDD41", "p_nom": 900.0, "carrier": "CCGT"},
              {"name": "Nuke B", "bus": "CCCC41", "p_nom": 1200.0, "carrier": "nuclear"}])
    out = _run(_tables(generators=g), "named_stations", refs={"named_station_buses": _station_ref()})
    assert _status(out, "named_station", "Big CCGT") == ["PASS"]
    assert _status(out, "named_station", "Nuke") == ["PASS"]
    assert _status(out, "named_station", "Small") == ["INFO"]  # optional and absent
    moved = g.assign(bus=["FFFF11", "CCCC41"])
    out = _run(_tables(generators=moved), "named_stations", refs={"named_station_buses": _station_ref()})
    assert _status(out, "named_station", "Big CCGT") == ["FAIL"]
    out = _run(_tables(generators=moved), "named_stations", refs={"named_station_buses": _station_ref()},
               named_stations={"exceptions": {"Big CCGT": ["FFFF11"]}})
    assert _status(out, "named_station", "Big CCGT") == ["INFO"]
    out = _run(_tables(generators=g.drop("Nuke B")), "named_stations", refs={"named_station_buses": _station_ref()})
    assert _status(out, "named_station", "Nuke") == ["FAIL"]  # required and absent


# ── check 4: electrolysis ──────────────────────────────────────────────────────

def _el_ref():
    return pd.DataFrame({"gsp": ["North GSP", "Direct(SPTL)"], "mw": [1000.0, 200.0],
                         "x": [200 * KM, 400 * KM], "y": [800 * KM, 400 * KM],
                         "known_boundaries": ["", "B4;B6"], "source": "FES"})


def test_electrolysis_behind_boundaries():
    links = pd.DataFrame({"bus0": ["AAAA41", "CCCC41"], "bus1": "H2", "p_nom": [1000.0, 200.0],
                          "carrier": "electrolysis", "interconnector_name": np.nan}, index=["e1", "e2"])
    out = _run(_tables(links=links), "electrolysis", refs={"fes_electrolysis_by_gsp": _el_ref()})
    assert _status(out, "electrolysis", "north of B4") == ["PASS"]
    assert _status(out, "electrolysis", "north of B6") == ["PASS"]
    south = links.assign(bus0="DDDD41")
    out = _run(_tables(links=south), "electrolysis", refs={"fes_electrolysis_by_gsp": _el_ref()})
    assert _status(out, "electrolysis", "north of B4") == ["FAIL"]
    assert _status(out, "electrolysis", "north of B6") == ["FAIL"]


def test_reference_range_widens_for_unknown_direct_rows():
    ref = _el_ref().assign(known_boundaries=["", "B6"])  # SPTL side of B4 unknown
    links = pd.DataFrame({"bus0": ["AAAA41"], "bus1": "H2", "p_nom": [1150.0], "carrier": "electrolysis",
                          "interconnector_name": np.nan}, index=["e1"])
    out = _run(_tables(links=links), "electrolysis", refs={"fes_electrolysis_by_gsp": ref})
    row = out[out["item"] == "north of B4"].iloc[0]
    assert row["status"] == "PASS" and str(row["reference"]).replace(",", "") == "1000-1200"


# ── check 5: LDES ──────────────────────────────────────────────────────────────

def _ldes_ref():
    return pd.DataFrame({"project": ["Glen PSH", "Coast LAES"], "technology": "pumped_hydro", "mw": [1000.0, 300.0],
                         "bus": ["BBBB41", "CCCC41"], "x": [600 * KM, 300 * KM], "y": [700 * KM, 400 * KM],
                         "region": ["North Scotland", "South Scotland"], "status": "new_build", "source": "TEC"})


def test_ldes_behind_boundaries_with_es5_residual():
    es5 = _es5(**{"Storage": (1000, 500, 0)})  # South Scotland: 300 named + 200 residual anywhere in it
    s = pd.DataFrame({"bus": ["BBBB41", "CCCC41"], "p_nom": [1000.0, 500.0], "carrier": "LAES"}, index=["s1", "s2"])
    out = _run(_tables(storage=s), "ldes", refs={"ldes_projects_2030": _ldes_ref()}, es5=es5)
    assert _status(out, "ldes", "north of B4") == ["PASS"]
    assert _status(out, "ldes", "north of B6") == ["PASS"]
    moved = s.assign(bus=["CCCC41", "CCCC41"])  # the Tayside case: PSH allowance on the wrong side of B4
    out = _run(_tables(storage=moved), "ldes", refs={"ldes_projects_2030": _ldes_ref()}, es5=es5)
    assert _status(out, "ldes", "north of B4") == ["FAIL"]
    assert _status(out, "ldes", "north of B6") == ["PASS"]
    assert "ldes_site" in set(out["check"])


# ── check 6: interconnectors ───────────────────────────────────────────────────

def _ic_ref():
    return pd.DataFrame({"interconnector": ["Link A", "Mares"], "match": ["^link a$", "^mares.*$"],
                         "landing_bus": ["DDDD41", "FFFF11"], "accepted_buses": ["DDDD41", "FFFF11"],
                         "capacity_mw": [1000.0, 750.0], "counterparty": "X", "counted_in_cp30": ["true", "true"],
                         "required": ["true", "false"], "source": "NESO register"})


def _es1(ic_mw):
    return pd.DataFrame([{"source_sheet": "ES1", "Connection": "Transmission", "Pathway": PATHWAY,
                          "Variable": "Capacity (MW)", "Category": "Interconnectors", "Type": "Interconnectors",
                          "SubType": "Interconnectors", "2030": ic_mw}])


def test_interconnectors_landing_capacity_and_total():
    buses_ext = {"bus0": ["EXT_External_X", "EXT_External_X"], "bus1": ["DDDD41", "DDDD41"]}
    links = pd.DataFrame({**buses_ext, "p_nom": [1000.0, 1000.0], "carrier": "DC",
                          "interconnector_name": ["Link A", "Link A"],
                          "interconnector_pair_id": ["A", "A"]}, index=["IC_IMPORT_A", "IC_EXPORT_A"])
    t = _tables(links=links)
    t.buses.loc["EXT_External_X"] = [0.0, 0.0, 400, "AC", "France"]
    out = _run(t, "interconnectors", refs={"interconnectors_2030": _ic_ref()}, es1=_es1(1000.0))
    assert _status(out, "interconnector", "Link A landing") == ["PASS"]
    assert _status(out, "interconnector", "Link A capacity") == ["PASS"]  # the pair counts once
    assert _status(out, "interconnector", "Mares landing") == ["WARN"]  # optional, absent
    assert _status(out, "interconnector", "CP30 total") == ["PASS"]
    links2 = links.assign(bus1="CCCC41", p_nom=1100.0)
    t2 = _tables(links=links2)
    t2.buses.loc["EXT_External_X"] = [0.0, 0.0, 400, "AC", "France"]
    out = _run(t2, "interconnectors", refs={"interconnectors_2030": _ic_ref()}, es1=_es1(1000.0))
    assert _status(out, "interconnector", "Link A landing") == ["FAIL"]
    assert _status(out, "interconnector", "Link A capacity") == ["WARN"]
    assert _status(out, "interconnector", "CP30 total") == ["FAIL"]


def test_interconnector_reference_takes_cp30_composition(tmp_path):
    """Capacities and which cables count come from the build's composition file at run time."""
    comp = tmp_path / "composition.csv"
    comp.write_text("# CP30 cables\n"
                    "name,action,cp30_mw,first_year,landing_point_gb,gb_bus,to_bus,counterparty_country,"
                    "international_location,international_latitude,international_longitude,gb_latitude,gb_longitude,"
                    "losses_percent,evidence\n"
                    "Link A,keep,1000,,,,,,,,,,,,nominal\n"
                    "Isle Link,exclude,0,,,,,,,,,,,,outside the count\n"
                    "Mares,add,750,2030,Bod,FFFF11,EXT,Ireland,M,53.0,-6.0,53.0,-3.5,2.5,step\n")
    landings = pd.DataFrame({"interconnector": ["Link A", "Isle", "Mares"],
                             "composition_name": ["Link A", "Isle Link", "Mares"],
                             "match": ["^link a$", "^isle.*$", "^mares.*$"], "landing_bus": ["DDDD41", "AAAA41", "FFFF11"],
                             "accepted_buses": ["DDDD41", "AAAA41", "FFFF11"], "counterparty": "X",
                             "register_mw": [1020.0, 74.0, 750.0], "source": "NESO register"})
    ref = lv.interconnector_reference(landings, comp, 2030)
    assert list(ref["capacity_mw"]) == [1000.0, 0.0, 750.0]
    assert list(ref["required"]) == [True, False, True] and list(ref["in_cp30_build"]) == [True, False, True]
    assert not lv.interconnector_reference(landings, comp, 2029)["required"].iloc[2]   # Mares not yet due
    links = pd.DataFrame({"bus0": ["EXT_External_X"], "bus1": ["DDDD41"], "p_nom": [1000.0], "carrier": "DC",
                          "interconnector_name": ["Link A"]}, index=["IC_A"])
    t = _tables(links=links)
    t.buses.loc["EXT_External_X"] = [0.0, 0.0, 400, "AC", "France"]
    out = _run(t, "interconnectors", refs={"interconnectors_reference": ref}, es1=_es1(1750.0))
    assert _status(out, "interconnector", "Link A capacity") == ["PASS"]
    assert _status(out, "interconnector", "Isle landing") == ["INFO"]        # excluded and absent
    assert _status(out, "interconnector", "Mares landing") == ["FAIL"]       # required by the composition
    assert _status(out, "interconnector", "CP30 total") == ["FAIL"]          # 1,000 against 1,750
    with pytest.raises(ValueError, match="disagree"):
        lv.interconnector_reference(landings.iloc[:2], comp, 2030)           # a composition row without a landing
    fallback = lv.interconnector_reference(landings, None, 2030)
    assert list(fallback["capacity_mw"]) == [1020.0, 74.0, 750.0] and fallback["required"].all()


def test_committed_interconnector_landings_match_the_repository_composition():
    path = Path("data/CP30/validation_references/interconnectors_2030.csv")
    comp = Path("data/interconnectors/cp30_interconnector_composition.csv")
    if not path.is_file() or not comp.is_file():
        pytest.skip("reference or composition file not available")
    ref = lv.interconnector_reference(pd.read_csv(path), comp, 2030).set_index("interconnector")
    assert ref.at["MaresConnect", "required"] and ref.at["MaresConnect", "capacity_mw"] == 750.0
    assert ref.at["MaresConnect", "accepted_buses"] == "BODE41"
    assert not ref.at["Isle of Man Interconnector", "required"]
    assert ref.loc[ref["counted_in_cp30"], "capacity_mw"].sum() == 12450.0     # CP30 ES1 2030


# ── check 7: demand ────────────────────────────────────────────────────────────

def _ed1(total_gwh):
    rows = [{"source_sheet": "ED1", "Data item": "GB Total System Demand - Minus electrolysis", "Unit": "GWh",
             "Pathway": PATHWAY, "2030": total_gwh},
            {"source_sheet": "ED1", "Data item": "GBFES Peak Customer Demand: Total Consumption plus Losses ",
             "Unit": "GW", "Pathway": PATHWAY, "2030": 0.5}]
    return pd.DataFrame(rows)


def _es1_peak(mw):
    return pd.DataFrame([{"source_sheet": "ES1", "Connection": "Both", "Pathway": PATHWAY, "Variable": "Demand (MW)",
                          "Category": "Demand", "Type": "Demand (MW) - Excluding Electrolysis", "2030": mw}])


def _demand_ref():
    return pd.DataFrame({"gsp": ["N", "M", "S"], "gwh": [10.0, 25.0, 65.0], "x": [200 * KM, 300 * KM, 300 * KM],
                         "y": [800 * KM, 400 * KM, 100 * KM], "known_boundaries": "", "source": "FES24"})


def test_demand_total_peak_and_shares_pass():
    # Loads: A 30 MWh (north of B4), C 70 MWh, D 200 MWh -> shares 10% / 33.3% north of B4 / B6.
    out = _run(_tables(), "demand", refs={"fes_demand_by_gsp": _demand_ref()},
               ed1=_ed1(0.3), es1=_es1_peak(200.0))
    assert _status(out, "demand", "annual demand") == ["PASS"]
    assert _status(out, "demand", "peak demand excl. electrolysis") == ["PASS"]
    assert _status(out, "demand", "ED1 GBFES") == ["INFO"]
    assert _status(out, "demand_share", "B4 remote side") == ["PASS"]
    assert _status(out, "demand_share", "B6 remote side") == ["PASS"]  # 33.3% against 35%: -1.7 pp


def test_demand_share_peak_and_total_fail():
    ref = _demand_ref().assign(gwh=[30.0, 0.0, 70.0])  # FES says 30% north of B4
    out = _run(_tables(), "demand", refs={"fes_demand_by_gsp": ref}, ed1=_ed1(0.31), es1=_es1_peak(150.0))
    assert _status(out, "demand", "annual demand") == ["FAIL"]  # 300 MWh against 310 MWh
    assert _status(out, "demand", "peak demand excl. electrolysis") == ["FAIL"]  # 200 against 150 MW
    assert _status(out, "demand_share", "B4 remote side") == ["FAIL"]  # 10% against 30%
    out = _run(_tables(), "demand", refs={"fes_demand_by_gsp": ref}, es1=_es1_peak(193.0))
    assert _status(out, "demand", "peak demand excl. electrolysis") == ["WARN"]  # +3.6%


def test_demand_share_reference_composes_named_loads_like_the_build():
    # GSP shares say 10% north of B4; a named load holding 20% of demand sits north of B4, so the
    # build places 0.8 x 10% + 20% = 28% there. The check must use the same composition.
    named = pd.DataFrame({"name": ["Big load"], "share_of_system_demand": [0.20], "x": [210 * KM], "y": [810 * KM],
                          "source": "FES24 ETYS type T"})
    refs = {"fes_demand_by_gsp": _demand_ref(), "transmission_named_loads": named}
    loads = pd.DataFrame({"bus": ["AAAA41", "CCCC41", "DDDD41"], "carrier": "electricity"}, index=["L_A", "L_C", "L_D"])
    placed = pd.DataFrame({"L_A": [28.0], "L_C": [20.0], "L_D": [52.0]})  # 28% / 48% north of B4 / B6
    out = _run(_tables(loads=loads, load_p=placed), "demand", refs=refs)
    row = out[out["item"] == "B4 remote side"].iloc[0]
    assert row["status"] == "PASS" and abs(float(row["reference"]) - 28.0) < 1e-6
    assert _status(out, "demand_share", "B6 remote side") == ["PASS"]  # 0.8 x 35% + 20% = 48%
    out = _run(_tables(loads=loads, load_p=placed), "demand", refs={"fes_demand_by_gsp": _demand_ref()},
               demand={"named_loads": False})
    assert _status(out, "demand_share", "B4 remote side") == ["FAIL"]  # GSP shares alone: 10%


# ── references derived at run time ────────────────────────────────────────────

def _direct():
    return pd.DataFrame({"node_id": ["Direct(SPTL)"], "licence": ["SPTL"], "x": [400 * KM], "y": [400 * KM],
                         "known_boundaries": ["B4;B6"], "region": ["South Scotland"], "source": "judgement"})


def _fes_rows():
    rows = [("Dem_BB003", "North", "NNNN_1", "AAAA41", 10.0), ("Dem_BB003", "Mid", "MMMM_1", "ZZZZ11", 25.0),
            ("Dem_BB003", "Direct(SPTL)", "Direct(SPTL)", "Direct(SPTL)", 5.0),
            ("Dem_BB003", "Gone", "QQQQ_1", "QQQQ11", 0.0), ("Gen_BB018", "North", "NNNN_1", "AAAA41", 7.0)]
    return pd.DataFrame([{"Building Block ID Number": b, "FES Pathway": "Holistic Transition", "GSP": g,
                          "GSP ID": gid, "Node ID": node, "2030": v} for b, g, gid, node, v in rows])


def test_fes_node_table_locates_bus_then_gsp_info_then_direct_point():
    lon, lat = _lonlat(300 * KM, 400 * KM)
    info = pd.DataFrame({"GSP ID": ["MMMM_1"], "Latitude": [lat], "Longitude": [lon]})
    t = lv.fes_node_table(_fes_rows(), "Dem_BB003", "Holistic Transition", 2030, _tables().buses, info, _direct())
    t = t.set_index("gsp")
    assert "Gone" not in t.index                                   # zero rows dropped
    assert t.at["North", "location_basis"] == "network_bus" and t.at["North", "y"] == 800 * KM
    assert t.at["Mid", "location_basis"] == "fes_gsp_info" and abs(t.at["Mid", "x"] - 300 * KM) < 1.0
    assert t.at["Direct(SPTL)", "known_boundaries"] == "B4;B6" and t.at["Direct(SPTL)", "licence_direct"] == "SPTL"


def test_derive_references_builds_fes_tables_and_named_loads(tmp_path, monkeypatch):
    fes = tmp_path / "fes.csv"
    _fes_rows().to_csv(fes, index=False)
    lon, lat = _lonlat(300 * KM, 400 * KM)
    info = tmp_path / "info.csv"
    pd.DataFrame({"GSP ID": ["MMMM_1"], "Latitude": [lat], "Longitude": [lon]}).to_csv(info, index=False)
    named = pd.DataFrame({"etys_id": ["T1", "T1", "T2"], "name": "x", "category": "industrial",
                          "node": ["AAAA41", "NOPE41", ""], "node_share": [0.5, 0.5, 1.0], "gwh": [5.0, 5.0, 10.0],
                          "share_of_system_demand": [0.05, 0.05, 0.1], "latitude": [57.0, 57.0, 51.0],
                          "longitude": [-4.0, -4.0, -1.0], "source": "build"})
    seen = {}

    def fake_reference(cfg, pathway, year, ed1, es1, log=None):
        seen.update(cfg=cfg, ed1=ed1)
        return named

    from scripts.demand import cp30_demand
    monkeypatch.setattr(cp30_demand, "named_load_reference", fake_reference)
    cfg = _config()
    sources = {"fes_data": str(fes), "gsp_info": str(info), "cp30_demand": {"enabled": True},
               "named_loads": "resources/demand/named.csv", "demand_lines": "resources/demand/lines.csv",
               "cp30_ed1": "ed1.csv"}
    refs, problems = lv.derive_references(sources, cfg, _tables().buses, _polygons(), _group_region(),
                                          {"fes_direct_row_locations": _direct()}, PATHWAY, 2030)
    assert not problems
    assert refs["fes_demand_by_gsp"]["gwh"].sum() == 40.0
    hm = refs["fes_regional_hydro_marine"].set_index(["technology", "region"])["mw"]
    assert hm[("hydro", "North Scotland")] == 7.0
    assert seen["cfg"]["named_loads"]["file"] == "resources/demand/named.csv"        # the declared input
    assert seen["cfg"]["national_lines_file"] == "resources/demand/lines.csv"
    nl = refs["transmission_named_loads"].set_index("node")
    assert nl.at["AAAA41", "location_basis"] == "network bus" and nl.at["AAAA41", "y"] == 800 * KM
    assert nl.at["NOPE41", "location_basis"] == "site coordinates"                   # node not in the network
    # Not enabled for the scenario: no named loads, the GSP shares alone.
    refs, _ = lv.derive_references({**sources, "cp30_demand": {}}, cfg, _tables().buses, _polygons(),
                                   _group_region(), {}, PATHWAY, 2030)
    assert "transmission_named_loads" not in refs


def test_offshore_landing_reference_crosswalk_wins_and_cfd_exclusions():
    cw = pd.DataFrame({"repd_ref_id": [1, 3], "site_name": ["Farm A", "Gone C"], "landing_bus": ["AAAA41", "CCCC41"],
                       "evidence": ["tec_register", "tec_register"], "source": ["TEC", "TEC"]})
    add = pd.DataFrame({"repd_ref_id": [1, 9], "site_name": ["Farm A old", "Pilot"], "landing_bus": ["BBBB41", "DDDD11"],
                        "evidence": ["site_location", "site_location"], "source": ["hand", "hand"]})
    cfd = pd.DataFrame({"repd_ref_id": [3, 1], "action": ["not_built_refill", "keep"],
                        "carrier": ["wind_offshore", "wind_offshore"]})
    ref = lv.offshore_landing_table(cw, add, cfd, "crosswalk.csv").set_index("repd_ref_id")
    assert ref.at[1, "landing_bus"] == "AAAA41" and ref.at[1, "source"].startswith("crosswalk.csv")
    assert ref.at[9, "landing_bus"] == "DDDD11"
    assert ref.at[3, "excluded_not_built"] and not ref.at[1, "excluded_not_built"] and ref["sourced"].all()


def test_unreferenced_named_farms_are_one_warn_row():
    g = _gen([{"name": "X", "bus": "BBBB41", "p_nom": 10.0, "carrier": "wind_offshore", "fleet_origin": "named",
               "repd_ref_id": 77, "source_site_name": "Pilot X"}])
    out = _run(_tables(generators=g), "offshore", refs={"offshore_landing_reference": _landing_ref()})
    assert _status(out, "offshore_named", "named farms with no landing reference") == ["WARN"]
    assert "Pilot X (REPD 77)" in out[out["item"] == "named farms with no landing reference"]["note"].iloc[0]


def test_demand_info_rows_from_the_etys_peak_table():
    lon_n, lat_n = _lonlat(200 * KM, 800 * KM)
    lon_s, lat_s = _lonlat(300 * KM, 100 * KM)
    etys = pd.DataFrame({"etys_id": ["N", "S", "U"], "type": "C", "peak_mw": [20.0, 60.0, 20.0],
                         "latitude": [lat_n, lat_s, np.nan], "longitude": [lon_n, lon_s, np.nan],
                         "location_basis": "GSP info", "source": "FES24 ETYS"})
    out = _run(_tables(), "demand", refs={"fes_demand_by_gsp": _demand_ref(), "fes24_etys_peak_by_gsp": etys})
    row = out[out["item"] == "B4 remote side vs FES24 ETYS winter peak"].iloc[0]
    assert row["status"] == "INFO" and abs(float(row["reference"]) - 20.0) < 1e-9   # unlocated counts in the total


# ── check 8: distributed ───────────────────────────────────────────────────────

def test_distributed_at_home_passes_and_away_fails():
    lon, lat = _lonlat(310 * KM, 110 * KM)  # inside the DDDD polygon
    rows = [{"name": f"s{i}", "bus": "DDDD11", "p_nom": 10.0, "carrier": "solar_pv", "lat": lat, "lon": lon}
            for i in range(10)]
    g = _gen(rows)
    out = _run(_tables(generators=g), "distributed")
    assert _status(out, "distributed", "solar_pv: distribution-size MW not at its GSP node") == ["PASS"]
    away = g.copy()
    away.loc[["s0", "s1", "s2"], "bus"] = "FFFF11"  # 30% at a substation that is not its GSP
    out = _run(_tables(generators=away), "distributed")
    assert _status(out, "distributed", "solar_pv: distribution-size MW not at its GSP node") == ["FAIL"]
    assert _status(out, "distributed", "substations with no demand") == ["WARN"]
    away.loc[["s1", "s2"], "bus"] = "DDDD11"  # back to 10% away ...
    away.loc[["s1"], "bus"] = "FFFF11"  # ... then 20%: WARN
    out = _run(_tables(generators=away), "distributed")
    assert _status(out, "distributed", "solar_pv: distribution-size MW not at its GSP node") == ["WARN"]


def _site(name, bus, mw, x=310, y=110, **extra):
    lon, lat = _lonlat(x * KM, y * KM)  # default: inside the DDDD (East England, _A) polygon
    return {"name": name, "bus": bus, "p_nom": mw, "carrier": "wind_onshore", "lat": lat, "lon": lon,
            "fleet_origin": "named", **extra}


def test_distributed_home_is_the_node_the_build_locator_gives():
    # Dem_per_node puts GSP DDDD_1's demand on FFFF11: the build places the GSP's distributed
    # plant there, so the check must count FFFF11 as home although its code is not the GSP id's.
    g = _gen([_site("a", "FFFF11", 50.0), _site("b", "DDDD11", 50.0)])
    dem = _dem_per_node([("DDDD_1", "FFFF11", 100.0)])
    out = _run(_tables(generators=g), "distributed", dem_per_node=dem)
    assert _status(out, "distributed", "wind_onshore: distribution-size MW not at its GSP node") == ["PASS"]
    out = _run(_tables(generators=g.assign(bus="AAAA41")), "distributed", dem_per_node=dem)
    assert _status(out, "distributed", "wind_onshore: distribution-size MW not at its GSP node") == ["FAIL"]


def test_distributed_judges_a_project_by_its_total_not_its_parts():
    # 60 + 60 MW of one REPD project is a 120 MW transmission-size project in England and Wales:
    # the build left it at its own bus, so it is not judged (Cowley Complex, Brochloch Rig).
    g = _gen([_site("p_CfD", "AAAA41", 60.0, repd_ref_id=1.0), _site("p_merchant", "AAAA41", 60.0, repd_ref_id=1.0),
              _site("small", "DDDD11", 40.0)])
    out = _run(_tables(generators=g), "distributed")
    row = out[out["item"] == "wind_onshore: distribution-size MW not at its GSP node"].iloc[0]
    assert row["status"] == "PASS" and float(row["reference"]) == pytest.approx(40.0)


def test_distributed_proxy_tranches_are_reported_not_judged():
    g = _gen([_site("named", "DDDD11", 40.0), _site("CP30_increment_wind_onshore_1", "AAAA41", 90.0,
                                                    fleet_origin="increment", data_source="CP30")])
    out = _run(_tables(generators=g), "distributed")
    assert _status(out, "distributed", "wind_onshore: distribution-size MW not at its GSP node") == ["PASS"]
    info = out[out["item"] == "wind_onshore: regional proxy tranches below the threshold (not judged)"].iloc[0]
    assert info["status"] == "INFO" and float(info["model"]) == pytest.approx(90.0)


def test_distributed_island_sites_are_at_home_on_the_island_network():
    codes = lv._island_codes(pd.Series([59.0, 57.0]), pd.Series([-3.0, -3.0]))
    assert "FINS" in codes.iloc[0] and not codes.iloc[1]


# ── check 9: stranded ──────────────────────────────────────────────────────────

def test_stranded_radial_pocket_warns():
    g = _gen([{"name": "w", "bus": "POCK1-", "p_nom": 120.0, "carrier": "wind_onshore"}])
    out = _run(_tables(generators=g), "stranded")
    assert _status(out, "stranded", "radial pockets") == ["PASS"]  # 120 <= 1.3 x 100
    out = _run(_tables(generators=g.assign(p_nom=200.0)), "stranded")
    assert _status(out, "stranded", "pocket at POCK1-") == ["WARN"]
    assert _status(out, "stranded", "radial pockets") == ["WARN"]


def test_stranded_ignores_relaxed_english_132kv_branches():
    # FFFF11 hangs off DDDD41 by a 400/132 kV transformer: relaxed in E&W (sub-220 kV).
    g = _gen([{"name": "w", "bus": "FFFF11", "p_nom": 5000.0, "carrier": "solar_pv"}])
    out = _run(_tables(generators=g), "stranded")
    assert _status(out, "stranded", "radial pockets") == ["PASS"]


# ── check 11: demand headroom ──────────────────────────────────────────────────

def _headroom_tables(electrolysis=None, load_mw=50.0, coupled=True, lines_extra=(), generators=None,
                     snapshots=None, load_p=None):
    """GRID41 (400 kV) -> GRID11 (132 kV) -> SITE1Q / SITE1R (132 kV, 132 MVA each; coupled or not), with a
    33 kV load busbar SITE3- under both (the BUMU/ABNE/BAIN shape). x on every branch."""
    buses = pd.DataFrame({"x": [300 * KM, 300 * KM, 310 * KM, 310 * KM, 310 * KM],
                          "y": [600 * KM, 600 * KM, 610 * KM, 610 * KM, 610 * KM],
                          "v_nom": [400.0, 132.0, 132.0, 132.0, 33.0], "carrier": "AC", "country": "GB"},
                         index=["GRID41", "GRID11", "SITE1Q", "SITE1R", "SITE3-"])
    lines = [("c_q", "GRID11", "SITE1Q", 132.0, 17.0), ("c_r", "GRID11", "SITE1R", 132.0, 17.0)]
    if coupled:
        lines.append(("coupler", "SITE1Q", "SITE1R", 9999.0, 0.0017))
    lines += list(lines_extra)
    lines = pd.DataFrame([dict(name=n, bus0=a, bus1=b, s_nom=s, x=x) for n, a, b, s, x in lines]).set_index("name")
    transformers = pd.DataFrame({"bus0": ["GRID41", "SITE1Q", "SITE1R"], "bus1": ["GRID11", "SITE3-", "SITE3-"],
                                 "s_nom": [2000.0, 90.0, 90.0], "x": [0.1, 0.2, 0.2]}, index=["sgt", "t_q", "t_r"])
    links = pd.DataFrame(columns=["bus0", "bus1", "p_nom", "carrier", "interconnector_name"])
    for bus, mw in (electrolysis or {}).items():
        links.loc[f"electrolysis_{bus}"] = [bus, "GB_H2", mw, "electrolysis", np.nan]
    loads = pd.DataFrame({"bus": ["SITE3-"], "carrier": "electricity"}, index=["L_SITE"])
    if load_p is None:
        load_p = pd.DataFrame({"L_SITE": [load_mw * 0.5, load_mw]})
    generators = generators if generators is not None else _gen([{"name": "none", "bus": "GRID41", "p_nom": 0.0,
                                                                  "carrier": "CCGT"}])
    return lv.NetworkTables(buses, generators, pd.DataFrame(columns=["bus", "p_nom", "carrier", "lat", "lon"]), links,
                            loads, lines, transformers, load_p, pd.Series(np.ones(len(load_p))), snapshots)


NO_FLOOR = {"min_bm_constraint_voltage_kv": 0.0}


def _headroom(tables, transmission=None):
    out = _run(tables, "demand_headroom", transmission=transmission or NO_FLOOR)
    return out[out["check"] == "demand_headroom"]


def test_demand_headroom_fails_electrolysis_behind_a_132kv_node():
    # BUMU: 390 MW at SITE1Q plus 50 MW of load on SITE3- behind 2 x 132 MVA.
    out = _headroom(_headroom_tables({"SITE1Q": 390.0}))
    row = out[out["item"].str.startswith("SITE1Q node")].iloc[0]
    assert row["status"] == "FAIL" and "SITE1Q|SITE1R" in row["item"]
    assert row["model"] == pytest.approx(440.0) and row["reference"] == pytest.approx(264.0, rel=1e-2)
    assert _status(out, "demand_headroom", "nodes whose peak demand") == ["FAIL"]


def test_demand_headroom_warns_between_the_thresholds_and_passes_below():
    out = _headroom(_headroom_tables({"SITE1Q": 240.0}))       # 290 / 264 = 1.10
    assert _status(out, "demand_headroom", "SITE1Q node") == ["WARN"]
    out = _headroom(_headroom_tables({"SITE1Q": 150.0}))       # 200 / 264
    assert _status(out, "demand_headroom", "nodes whose peak demand") == ["PASS"]
    assert len(out) == 1


def test_demand_headroom_judges_uncoupled_busbars_on_their_own_circuit():
    # BAIN: SITE1Q and SITE1R have no coupler. 100 MW at SITE1Q plus SITE3-'s 120 MW electrolyser and 25 MW
    # load against 2 x 132 MVA is 0.93 x summed, but SITE1Q's circuit carries most of its own and half of SITE3-'s.
    out = _headroom(_headroom_tables({"SITE1Q": 100.0, "SITE3-": 120.0}, load_mw=25.0, coupled=False))
    row = out[out["item"].str.startswith("SITE1Q node")].iloc[0]
    assert row["status"] in {"WARN", "FAIL"} and "first to bind c_q" in row["note"]
    assert not out["item"].str.startswith("SITE1R node").any()


def test_demand_headroom_fails_a_load_only_shedding_can_serve_unless_firm_generation_covers_it():
    # 290 MW of load (no electrolysis) against 264: ratio 1.10 (a WARN by ratio), but only shedding serves it.
    out = _headroom(_headroom_tables(load_mw=290.0))
    row = out[out["item"].str.startswith("SITE1Q node")].iloc[0]
    assert row["status"] == "FAIL" and "only load shedding can serve it" in row["note"]
    firm = _gen([{"name": "chp", "bus": "SITE3-", "p_nom": 40.0, "carrier": "CHP"}])
    out = _headroom(_headroom_tables(load_mw=290.0, generators=firm))
    assert _status(out, "demand_headroom", "SITE1Q node") == ["WARN"]


def test_demand_headroom_reads_rating_overrides_and_seasonal_ratings():
    # ETYS rates one circuit at 1 MVA; the BM's line_rating_overrides restore it (ELDE-JOHN).
    t = _headroom_tables({"SITE1Q": 150.0})
    t.lines.loc["c_q", "s_nom"] = 1.0
    t.lines.loc["c_r", "s_nom"] = 1.0
    assert _status(_headroom(t), "demand_headroom", "nodes whose peak demand") == ["FAIL"]
    fixed = dict(NO_FLOOR, line_rating_overrides={"c_q": 132.0, "c_r": 132.0})
    assert _status(_headroom(t, fixed), "demand_headroom", "nodes whose peak demand") == ["PASS"]
    # 230 / 264 passes at winter ratings and fails at a summer rating of 106 MVA (May).
    t = _headroom_tables({"SITE1Q": 180.0}, snapshots=pd.DatetimeIndex(["2030-01-15 18:00", "2030-05-15 12:00"]),
                         load_p=pd.DataFrame({"L_SITE": [50.0, 50.0]}))
    for col, value in (("s_nom_winter", 132.0), ("s_nom_spring", 123.0), ("s_nom_summer", 106.0),
                       ("s_nom_autumn", 123.0)):
        t.lines[col] = value
    t.lines.loc["coupler", ["s_nom_winter", "s_nom_spring", "s_nom_summer", "s_nom_autumn"]] = 9999.0
    seasonal = dict(NO_FLOOR, seasonal_ratings={"enabled": True})
    row = _headroom(t, seasonal)
    row = row[row["item"].str.startswith("SITE1Q node")].iloc[0]
    assert row["status"] == "WARN" and "in May" in row["note"]
    no_season = dict(NO_FLOOR, seasonal_ratings={"enabled": False})
    assert _status(_headroom(t, no_season), "demand_headroom", "nodes whose peak demand") == ["PASS"]


def test_demand_headroom_ignores_branches_the_bm_relaxes():
    # E&W sub-220 kV branches are relaxed in the BM: no 132 kV node binds.
    out = _headroom(_headroom_tables({"SITE1Q": 900.0}), {"min_bm_constraint_voltage_kv": 220.0,
                                                           "bm_voltage_floor": {"scottish_transmission_constrained": False}})
    assert _status(out, "demand_headroom", "nodes whose peak demand") == ["PASS"]


# ── check 10: exposure ─────────────────────────────────────────────────────────

def test_exposure_is_info_with_sides():
    g = _gen([{"name": "w", "bus": "AAAA41", "p_nom": 700.0, "carrier": "wind_onshore"}])
    out = _run(_tables(generators=g), "exposure")
    assert set(out["status"]) == {"INFO"}
    row = out[out["item"] == "B4 remote side: generation MW"].iloc[0]
    assert row["model"] == 700.0
    assert out[out["item"] == "B4 remote side: mean demand MW"].iloc[0]["model"] == 15.0


def test_failures_and_broken_check_reported():
    table = pd.DataFrame([lv._row("x", "a", status="FAIL"), lv._row("x", "b", status="PASS")], columns=lv.COLUMNS)
    assert len(lv.locational_failures(table)) == 1
    out = _run(_tables(), "electrolysis", refs={"fes_electrolysis_by_gsp": pd.DataFrame({"mw": [1.0]})})
    assert "FAIL" in set(out["status"])  # a reference without coordinates raises -> FAIL row, not a crash


# ── live network ───────────────────────────────────────────────────────────────

LIVE_NETWORK = Path(os.environ.get(
    "CP30_LOCVAL_NETWORK", "resources/network/CP30_FFR_2030.nc"))
LIVE_CP30 = Path(os.environ.get("CP30_LOCVAL_CP30_DIR", "resources/CP30"))


@pytest.fixture(scope="module")
def live_table():
    if not LIVE_NETWORK.is_file() or not (LIVE_CP30 / "CP30_ES5_regional.csv").is_file():
        pytest.skip(f"built CP30 network not available ({LIVE_NETWORK})")
    tables = lv.NetworkTables.from_netcdf(LIVE_NETWORK)
    cfg = lv.DEFAULT_CONFIG
    sources = {"fes_data": "resources/FES/FES_2024_data.csv", "gsp_info": cfg["fes_reference"]["gsp_info_file"],
               "offshore_crosswalk": "data/network/ETYS/repd_offshore_landing_crosswalk.csv",
               "cfd_contracts": "data/market/lccc/checked_cfd_contracts.csv",
               "named_loads": "resources/demand/cp30_2030_named_large_loads.csv",
               "demand_lines": "resources/demand/fes24_ht_2030_demand_lines.csv",
               "etys_peak": "resources/demand/fes24_etys_ht_2030_peak_by_gsp.csv",
               "ic_composition": "data/interconnectors/cp30_interconnector_composition.csv",
               "cp30_demand": {"enabled": True}, "cp30_ed1": str(LIVE_CP30 / "CP30_ED1.csv"),
               "cp30_es1": str(LIVE_CP30 / "CP30_ES1.csv")}
    sources = {k: v for k, v in sources.items() if not isinstance(v, str) or not v.startswith("resources/")
               or Path(v).is_file()}
    if "named_loads" not in sources:
        sources["cp30_demand"] = {}
    return lv.run_locational_checks(
        tables, {}, es5=pd.read_csv(LIVE_CP30 / "CP30_ES5_regional.csv"), ed1=pd.read_csv(LIVE_CP30 / "CP30_ED1.csv"),
        es1=pd.read_csv(LIVE_CP30 / "CP30_ES1.csv"), pathway=PATHWAY, year=2030, sources=sources)


def test_live_network_runs_every_check(live_table):
    assert not (live_table["item"] == "check raised an exception").any(), \
        live_table[live_table["item"] == "check raised an exception"].to_string()
    for check in ("es5_regional", "offshore_proxy", "named_station", "electrolysis", "ldes", "interconnector",
                  "demand", "demand_share", "distributed", "stranded", "exposure", "demand_headroom"):
        assert check in set(live_table["check"]), check


@pytest.mark.skipif(os.environ.get("CP30_LOCVAL_EXPECT_KNOWN_ISSUES") != "1",
                    reason="set CP30_LOCVAL_EXPECT_KNOWN_ISSUES=1 to check the pre-fix network of 14 Oct 2026")
def test_live_network_flags_the_known_issues(live_table):
    t = live_table

    def status(check, item):
        return set(t[(t["check"] == check) & (t["item"] == item)]["status"])

    assert status("offshore_proxy", "B4: proxy landing at BLHI4-") == {"FAIL"}  # Blackhillock
    assert status("offshore_region", "tag South Scotland | landing BLHI4- in North Scotland") == {"FAIL"}
    assert status("electrolysis", "north of B4") == {"FAIL"} and status("electrolysis", "north of B6") == {"FAIL"}
    assert status("ldes", "north of B2") == {"FAIL"} and status("ldes", "north of B4") == {"FAIL"}
    assert status("es5_regional", "non_battery_storage | North Scotland") == {"FAIL"}  # Cruachan's region
    assert status("demand_share", "B8 remote side") & {"WARN", "FAIL"}
    assert status("distributed", "solar_pv: distribution-size MW not at its GSP node") & {"WARN", "FAIL"}
    assert status("interconnector", "MaresConnect landing") == {"FAIL"}  # required by the CP30 composition
    assert status("stranded", "radial pockets above connection + margin") == {"WARN"}
