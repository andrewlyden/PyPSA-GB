"""CP30 2030 demand build (demand audit fixes, 15 Oct 2026): scripts/demand/cp30_demand.py
and the same-site capacity redirect in scripts/demand/load.py.

Fix 1: GSP shares from FES 2024, Direct(TO) rows kept within their TO.
Fix 2: named large loads at their nodes, losses pro rata, only the residual scaled.
Fix 3: demand above a capped <=132 kV bus's capacity goes to its own site's supergrid.
Fix 4: component shapes and the base exponent: energy exact, peak on target.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import pytest
import yaml

from scripts.demand import cp30_demand as C
from scripts.demand import load as L
from config.config_loader import load_layer

LOG = logging.getLogger("test_cp30_demand_build")
ROOT = Path(__file__).resolve().parents[2]


# ───────────────────────────── helpers ─────────────────────────────

def _fes(rows):
    """FES building-block rows: (block, area, gsp, value_2030)."""
    return pd.DataFrame([{"Building Block ID Number": b, "FES Pathway": "Holistic Transition",
                          "DNO License Area": a, "GSP": g, "2030": v} for b, a, g, v in rows])


def _lines(**over):
    lines = {"data_centres_transmission_gwh": 100.0, "data_centres_distribution_gwh": 50.0,
             "direct_transmission_customers_gwh": 60.0, "residential_evs_gwh": 60.0,
             "commercial_evs_gwh": 40.0, "residential_evs_at_peak_gw": 0.010,
             "commercial_evs_at_peak_gw": 0.010, "residential_heat_pumps_at_peak_gw": 0.01,
             "commercial_heat_pumps_at_peak_gw": 0.01}
    lines.update(over)
    return lines


def _network(buses, transformers=(), lines=()):
    n = pypsa.Network()
    for name, (x, y, v) in buses.items():
        n.add("Bus", name, x=x, y=y, v_nom=v)
    for i, (b0, b1, s) in enumerate(transformers):
        n.add("Transformer", f"t{i}", bus0=b0, bus1=b1, s_nom=s, x=0.1)
    for i, (b0, b1, s) in enumerate(lines):
        n.add("Line", f"l{i}", bus0=b0, bus1=b1, s_nom=s, x=0.1)
    return n


# ───────────────────────────── fix 1: FES 2024 shares ─────────────────────────────

def test_direct_rows_are_spread_within_their_own_to_area():
    fes = _fes([("Dem_BB003", "SPEN", "Glasgow", 300.0), ("Dem_BB003", "SPEN", "Edinburgh", 100.0),
                ("Dem_BB003", "SPTL", "Direct(SPTL)", 40.0), ("Dem_BB003", "UKPN London", "London", 600.0)])
    out = C.fes_gsp_blocks(fes, "Holistic Transition", 2030, ["Dem_BB003"], {"Direct(SPTL)": "SPEN"})
    assert "Direct(SPTL)" not in out.index
    assert out.sum() == pytest.approx(1040.0)
    assert out["London"] == pytest.approx(600.0)            # nothing leaks south
    assert out["Glasgow"] == pytest.approx(330.0)           # 3/4 of the 40
    assert out["Edinburgh"] == pytest.approx(110.0)


def test_unmapped_direct_row_is_left_for_the_split_to_report():
    fes = _fes([("Dem_BB003", "SPEN", "Glasgow", 300.0), ("Dem_BB003", "SPTL", "Direct(SPTL)", 40.0)])
    out = C.fes_gsp_blocks(fes, "Holistic Transition", 2030, ["Dem_BB003"], {})
    assert out["Direct(SPTL)"] == pytest.approx(40.0)


def test_config_takes_fes_2024_by_default():
    cfg = C.resolve_config({"enabled": True})
    assert cfg["spatial"]["FES_year"] == 2024
    assert cfg["capacity_redistribution"] == "same_site"
    assert C.is_enabled(cfg) and not C.is_enabled({})


# ───────────────────────────── fix 2: named loads ─────────────────────────────

def _named():
    return pd.DataFrame({
        "category": ["data_centre", "data_centre", "rail", "steel_eaf"],
        "peak_mw_ht_2030_31": [300.0, 100.0, 50.0, 150.0],
        "nodes": ["DIDC41:1", "IVER11:1", "ECLA41:1", "MAGA21:1"],
    }, index=pd.Index(["DC1", "DC2", "R1", "PT"], name="etys_id"))


def test_named_loads_take_their_group_line_pro_rata_to_peak():
    e = C.named_load_energies(_named(), _lines(), C.DEFAULT_CONFIG["named_loads"]["group_lines"])
    assert e[["DC1", "DC2"]].sum() == pytest.approx(100.0)
    assert e["DC1"] == pytest.approx(75.0)
    assert e[["R1", "PT"]].sum() == pytest.approx(60.0)
    assert e["PT"] == pytest.approx(45.0)


def test_energy_table_scales_only_the_residual_and_spreads_losses_pro_rata():
    fes = _fes([("Dem_BB003", "A", "North", 400.0), ("Dem_BB003", "B", "South", 600.0)])
    cfg = C.resolve_config({"enabled": True})
    out = C.build_energy_table(fes, cfg, 2030, system_total_gwh=1100.0, consumer_total_gwh=1000.0,
                               named=_named(), lines=_lines())
    lf = 1.1
    assert out["accounting"]["loss_factor"] == pytest.approx(lf)
    assert out["named_gwh"].sum() == pytest.approx(160.0 * lf)
    assert out["gsp_gwh"].sum() + out["named_gwh"].sum() == pytest.approx(1100.0)
    assert out["gsp_gwh"]["North"] / out["gsp_gwh"]["South"] == pytest.approx(400 / 600)
    assert set(out["named_gwh"].index) == {C.NAMED_PREFIX + i for i in ("DC1", "DC2", "R1", "PT")}


def test_named_loads_are_placed_on_their_declared_node(monkeypatch, tmp_path):
    meta = tmp_path / "gsp_info.csv"
    pd.DataFrame({"GSP ID": ["ABHA1"], "Name": ["Abham"], "Latitude": [50.47], "Longitude": [-3.72]}).to_csv(meta, index=False)
    dem_per_node = pd.DataFrame({"GSP Id": ["ABHA1"], "Node Id": ["ABHA11"], "GSP Group ID": ["_L"],
                                 "Dem as % of demand within the GSP Group ID per each node": [1.0]})
    monkeypatch.setattr(L, "resolve_gsp_metadata_path", lambda *a, **k: meta)
    monkeypatch.setattr(L.pd, "read_excel", lambda *a, **k: dem_per_node)
    monkeypatch.setattr(L, "load_gsp_location_overrides", lambda *a, **k: {})
    n = _network({"ABHA11": (280000, 70000, 132), "DIDC41": (450000, 190000, 400), "IVER11": (505000, 182000, 132)},
                 lines=[("ABHA11", "DIDC41", 1000), ("DIDC41", "IVER11", 1000)])
    idx = pd.date_range("2030-01-01", periods=2, freq="h")
    ts = pd.DataFrame({"Abham": [50.0, 55.0], "NAMED::DC1": [30.0, 30.0], "NAMED::GONE": [10.0, 10.0]}, index=idx)
    annual = pd.DataFrame({"demand": ts.sum() / 1e3})
    shares = {"NAMED::DC1": {"nodes": [("DIDC41", 1.0)], "lonlat": None},
              # a node missing from the network falls back to its own site's bus
              "NAMED::GONE": {"nodes": [("IVER12", 1.0)], "lonlat": None}}
    _, node_ts = L.distribute_future_fes_demand_to_etys_nodes(
        annual, ts, n, LOG, fes_year=2024, direct_node_shares=shares, capacity_redistribution="same_site")
    assert node_ts["DIDC41"].tolist() == pytest.approx([30.0, 30.0])
    assert node_ts["IVER11"].tolist() == pytest.approx([10.0, 10.0])
    assert node_ts["ABHA11"].tolist() == pytest.approx([50.0, 55.0])


FIXTURES = ROOT / "tests/fixtures/cp30_demand"


def test_reviewed_named_loads_tables_are_complete():
    """The reviewed tables of 15 Oct 2026 (regression fixtures of rule extract_cp30_demand_inputs)."""
    named = C.read_named_loads(FIXTURES / "cp30_2030_named_large_loads.csv")
    assert set(named["category"]) >= {"data_centre", "rail", "industrial", "steel_eaf", "offshore_electrification"}
    dc = named[named["category"] == "data_centre"]
    assert dc["peak_mw_ht_2030_31"].sum() == pytest.approx(1540.0)
    for etys_id, row in named.iterrows():
        shares = C.parse_node_shares(row["nodes"])
        assert shares or pd.notna(row["latitude"]), etys_id
        if shares:
            assert sum(s for _, s in shares) == pytest.approx(1.0, abs=1e-5), etys_id
    lines = C.read_national_lines(FIXTURES / "fes24_ht_2030_demand_lines.csv")
    for key in ("data_centres_transmission_gwh", "data_centres_distribution_gwh", "direct_transmission_customers_gwh",
                "residential_evs_gwh", "commercial_evs_gwh", "residential_evs_at_peak_gw", "commercial_evs_at_peak_gw",
                "residential_heat_pumps_at_peak_gw", "commercial_heat_pumps_at_peak_gw", "peak_total_gw"):
        assert key in lines
    assert lines["data_centres_transmission_gwh"] == pytest.approx(13185.0)
    assert lines["peak_total_gw"] == pytest.approx(62.357)


def test_node_choices_file_is_a_declared_data_source():
    from scripts.demand import extract_cp30_demand_inputs as X

    choices = X.read_node_choices(ROOT / X.NODE_CHOICES)
    assert all(k.startswith("B_EXTRA_") for k in choices)
    assert choices["B_EXTRA_202"][0] == "MAGA21"                       # Port Talbot at Margam 275 kV
    assert choices["B_EXTRA_77"][0] == "" and choices["B_EXTRA_77"][2] is None   # FLOP centroid
    assert all(ev for _, ev, _, _ in choices.values())


def test_node_choices_reject_a_row_without_evidence(tmp_path):
    from scripts.demand import extract_cp30_demand_inputs as X

    f = tmp_path / "nodes.csv"
    f.write_text("# header\netys_id,nodes,node_evidence,latitude,longitude\nB_EXTRA_1,AAAA41,,51.0,-1.0\n")
    with pytest.raises(ValueError, match="node_evidence"):
        X.read_node_choices(f)


def test_extract_rule_feeds_the_demand_build():
    """build_base_demand reads the extraction rule's outputs: the config defaults are its outputs."""
    rules = (ROOT / "rules/demand.smk").read_text(encoding="utf-8")
    assert "rule extract_cp30_demand_inputs:" in rules
    assert 'node_choices="data/demand/cp30_named_load_nodes.csv"' in rules
    for key, path in (("CP30_NAMED_LOADS_FILE", C.DEFAULT_CONFIG["named_loads"]["file"]),
                      ("CP30_DEMAND_LINES_FILE", C.DEFAULT_CONFIG["national_lines_file"])):
        assert f'{key} = f"{{resources_path}}/{path.split("/", 1)[1]}"' in rules, key
        assert path.startswith("resources/demand/")
    assert not (ROOT / "data/demand/cp30_2030_named_large_loads.csv").exists()   # no committed copy to drift


@pytest.mark.skipif(not (ROOT / "data/network/ETYS/Regional breakdown of FES24 data.xlsx").is_file()
                    or not (ROOT / "data/FES/Future Energy Scenarios 2024 Data Workbook_V006_0.xlsx").is_file(),
                    reason="FES 2024 workbooks not available")
def test_extraction_reproduces_the_reviewed_tables(tmp_path, monkeypatch):
    """The rule's script, run on the repository workbooks, gives the reviewed tables (line endings aside)."""
    from scripts.demand import extract_cp30_demand_inputs as X

    monkeypatch.chdir(ROOT)
    X.run(X.ETYS_BREAKDOWN, X.GB_NETWORK, X.NODE_CHOICES, X.FES24_WORKBOOK, tmp_path / "named.csv",
          tmp_path / "lines.csv", tmp_path / "peak.csv", log=LOG)
    for out, fixture in (("named.csv", "cp30_2030_named_large_loads.csv"), ("lines.csv", "fes24_ht_2030_demand_lines.csv")):
        got = (tmp_path / out).read_text(encoding="utf-8").splitlines()
        want = (FIXTURES / fixture).read_text(encoding="utf-8").splitlines()
        assert got == want, fixture
    peak = pd.read_csv(tmp_path / "peak.csv")
    assert peak.loc[peak["latitude"].isna(), "peak_mw"].sum() == 0.0      # every MW located
    assert peak["peak_mw"].sum() == pytest.approx(62829.669, abs=0.01)


def test_named_load_reference_is_the_builds_own_placement(tmp_path, monkeypatch):
    """The locational check's named-loads reference uses the build's split and loss factor."""
    named = _named().reset_index().assign(latitude=51.5, longitude=-1.0, name="x")
    named.loc[named["etys_id"] == "PT", "nodes"] = "MAGA21:0.5;MAGB21:0.5"
    named.to_csv(tmp_path / "named.csv", index=False)
    pd.DataFrame({"key": list(_lines()), "value": list(_lines().values())}).to_csv(tmp_path / "lines.csv", index=False)
    ed1 = pd.DataFrame([{"Pathway": "P", "Data item": item, "Unit": "GWh", "Annual/ Peak/ Flex": "Annual", "2030": v}
                        for item, v in (("Total Residential", 500.0), ("Total Industrial and Commercial", 400.0),
                                        ("Total Road Transport", 100.0), ("Residential Heat Pump Flex", 10.0),
                                        ("Residential District Heat Flex", 5.0),
                                        ("GB Total System Demand - Minus electrolysis", 1100.0))])
    ed1.to_csv(tmp_path / "ed1.csv", index=False)
    cfg = {"enabled": True, "named_loads": {"file": str(tmp_path / "named.csv")},
           "national_lines_file": str(tmp_path / "lines.csv"), "ed1_component_pathway": "P"}
    ref = C.named_load_reference(cfg, "P", 2030, tmp_path / "ed1.csv")
    assert ref.attrs["loss_factor"] == pytest.approx(1.1)
    table = C.build_energy_table(_fes([("Dem_BB003", "A", "North", 1.0)]), C.resolve_config(cfg), 2030, 1100.0,
                                 1000.0, C.read_named_loads(tmp_path / "named.csv"), _lines())
    assert ref["gwh"].sum() == pytest.approx(table["named_gwh"].sum())
    assert ref["share_of_system_demand"].sum() == pytest.approx(table["named_gwh"].sum() / 1100.0)
    pt = ref[ref["etys_id"] == "PT"]
    assert list(pt["node"]) == ["MAGA21", "MAGB21"] and pt["gwh"].sum() == pytest.approx(45.0 * 1.1)
    assert C.named_load_reference({**cfg, "named_loads": {"enabled": False}}, "P", 2030, tmp_path / "ed1.csv").empty


# ───────────────────────────── fix 3: same-site redirect ─────────────────────────────

def _capped_site():
    return _network(
        {"AAAA41": (400000, 300000, 400), "AAAA11": (400000, 300000, 132), "AAAA31": (400000, 300000, 33),
         "BBBB41": (500000, 300000, 400), "BBBB11": (500000, 300000, 132)},
        transformers=[("AAAA41", "AAAA11", 100.0), ("AAAA11", "AAAA31", 50.0), ("BBBB41", "BBBB11", 1000.0)],
        lines=[("AAAA41", "BBBB41", 5000.0)])


def test_excess_goes_to_the_same_sites_supergrid_not_pro_rata():
    n = _capped_site()
    idx = pd.date_range("2030-01-01", periods=3, freq="h")
    ts = {"AAAA11": pd.Series([50.0, 150.0, 200.0], index=idx), "BBBB11": pd.Series([300.0] * 3, index=idx)}
    demand = {k: float(v.sum() / 1e3) for k, v in ts.items()}
    demand, ts, moves = L.redirect_capped_demand_to_supergrid(demand, ts, n, LOG, safety_margin=0.9)
    assert ts["AAAA11"].tolist() == pytest.approx([50.0, 90.0, 90.0])     # capped at 0.9 x 100
    assert ts["AAAA41"].tolist() == pytest.approx([0.0, 60.0, 110.0])     # only the excess moves
    assert ts["BBBB11"].tolist() == pytest.approx([300.0] * 3)            # no pro-rata spread
    assert sum(v.sum() for v in ts.values()) == pytest.approx(50 + 150 + 200 + 900)
    assert set(moves["to_bus"]) == {"AAAA41"}
    assert moves["gwh"].sum() == pytest.approx(0.170)


def test_capped_lower_voltage_bus_goes_up_to_the_supergrid_in_one_pass():
    n = _capped_site()
    idx = pd.date_range("2030-01-01", periods=2, freq="h")
    ts = {"AAAA31": pd.Series([80.0, 80.0], index=idx)}   # 33 kV board fed by a 50 MVA transformer
    _, ts, moves = L.redirect_capped_demand_to_supergrid({"AAAA31": 0.16}, ts, n, LOG, safety_margin=0.9)
    assert ts["AAAA31"].max() == pytest.approx(45.0)
    assert ts["AAAA41"].tolist() == pytest.approx([35.0, 35.0])          # skips the 132 kV board
    assert "AAAA11" not in ts or ts["AAAA11"].sum() == pytest.approx(0.0)


def test_supergrid_targets_falls_back_to_same_site_code_then_nearest():
    n = _network({"CCCC11": (100000, 100000, 132), "CCCC21": (100000, 100000, 275),
                  "DDDD11": (300000, 100000, 132), "EEEE41": (310000, 100000, 400)})
    assert L.supergrid_targets("CCCC11", n) == [("CCCC21", 1.0)]
    assert L.supergrid_targets("DDDD11", n) == [("EEEE41", 1.0)]


def test_pro_rata_stays_the_default_for_other_scenarios():
    import inspect
    sig = inspect.signature(L.distribute_future_fes_demand_to_etys_nodes)
    assert sig.parameters["capacity_redistribution"].default == "pro_rata"
    with pytest.raises(ValueError, match="capacity_redistribution"):
        L.distribute_future_fes_demand_to_etys_nodes(pd.DataFrame({"d": []}), pd.DataFrame(), _capped_site(), LOG,
                                                     capacity_redistribution="nearest")


# ───────────────────────────── fix 4: shape and peak ─────────────────────────────

def _hourly_year():
    idx = pd.date_range("2030-01-01", "2030-12-31 23:00", freq="h")
    doy = idx.dayofyear.to_numpy()
    hour = idx.hour.to_numpy()
    base = 1.0 + 0.25 * np.cos(2 * np.pi * (doy - 15) / 365) + 0.2 * np.sin(np.pi * np.clip(hour - 6, 0, 14) / 14)
    temp = pd.Series(8.0 - 7.0 * np.cos(2 * np.pi * (doy - 15) / 365) + 2.0 * np.sin(2 * np.pi * (hour - 9) / 24),
                     index=idx)
    return idx, pd.Series(base, index=idx), temp


def test_transform_base_keeps_energy_and_sharpens_with_alpha():
    _, base, _ = _hourly_year()
    for a in (0.5, 1.0, 2.0):
        t = C.transform_base(base.to_numpy(), a)
        assert t.mean() == pytest.approx(1.0)
    assert C.transform_base(base.to_numpy(), 2.0).max() > C.transform_base(base.to_numpy(), 1.0).max()


def test_calibrated_profiles_hit_the_peak_and_keep_every_gsps_energy():
    idx, base, temp = _hourly_year()
    cfg = C.resolve_config({"enabled": True})
    lines = _lines(residential_evs_at_peak_gw=0.002, commercial_evs_at_peak_gw=0.003)
    sh = C.national_shapes(idx, temp, cfg, lines, ev_gwh=100.0, loss_factor=1.1,
                           ev_residential_share=0.6, ev_depot_share=0.1)
    for col in sh["shapes"]:
        assert (sh["shapes"][col] * sh["dt"]).sum() == pytest.approx(1.0)
    gsp = pd.Series({"North": 4000.0, "South": 6000.0})
    comps = pd.DataFrame({"electric_vehicles": [40.0, 70.0], "heat_pumps": [60.0, 50.0],
                          "district_heat": [10.0, 20.0], "data_centres": [5.0, 45.0]}, index=gsp.index)
    named = pd.Series({"NAMED::DC": 87.6})
    natural = C.build_profiles(base, gsp, comps, named, sh["shapes"], sh["dt"], None)["record"]["peak_mw"]
    target = natural * 1.05
    out = C.build_profiles(base, gsp, comps, named, sh["shapes"], sh["dt"], target)
    p = out["p_set"]
    assert p.sum(axis=1).max() == pytest.approx(target, abs=0.05)
    assert out["record"]["base_exponent"] > 1.0
    assert (p[["North", "South"]].sum() / 1e3).tolist() == pytest.approx([4000.0, 6000.0])
    assert p["NAMED::DC"].nunique() == 1 and p["NAMED::DC"].iloc[0] == pytest.approx(10.0)
    assert (p >= 0).all().all()


def test_ev_smart_share_puts_ev_load_at_the_fes_at_peak_figure():
    idx, _, temp = _hourly_year()
    cfg = C.resolve_config({"enabled": True})
    lines = _lines(residential_evs_at_peak_gw=0.004, commercial_evs_at_peak_gw=0.006)
    sh = C.national_shapes(idx, temp, cfg, lines, ev_gwh=100.0, loss_factor=1.1,
                           ev_residential_share=0.6, ev_depot_share=0.1)
    rec = sh["record"]
    assert 0.0 < rec["ev_smart_share_of_residential"] < cfg["shape"]["ev_smart_max_share"]
    assert rec["ev_at_peak_mw"] == pytest.approx(10.0, rel=1e-6)          # consumer basis, MW


def test_heat_pump_shape_is_higher_in_the_cold():
    idx, _, temp = _hourly_year()
    sh = C.national_shapes(idx, temp, C.resolve_config({"enabled": True}), _lines(), 100.0, 1.1, 0.6, 0.1)
    hp = sh["shapes"]["heat_pumps"]
    assert hp[idx.month == 1].mean() > 3 * hp[idx.month == 7].mean()


def test_peak_target_resolves_es1_ed1_and_numbers():
    es1 = pd.DataFrame({"Pathway": ["Further Flex and Renewables"], "Type": ["Demand (MW) - Excluding Electrolysis"],
                        "2030": [56284.67]})
    ed1 = pd.DataFrame({"Pathway": ["Further Flex and Renewables"],
                        "Data item": ["GBFES Peak Customer Demand: Total Consumption plus Losses "], "2030": [62.357]})
    assert C.peak_target_mw("es1", "Further Flex and Renewables", 2030, es1, ed1)[0] == pytest.approx(56284.67)
    assert C.peak_target_mw("ed1", "Further Flex and Renewables", 2030, es1, ed1)[0] == pytest.approx(62357.0)
    assert C.peak_target_mw(60000, "x", 2030, None, None)[0] == 60000.0
    with pytest.raises(ValueError):
        C.peak_target_mw("acs", "x", 2030, es1, ed1)


def test_components_capped_within_their_gsp_keep_totals():
    fes = _fes([("Dem_BB006", "A", "Small", 90.0), ("Dem_BB006", "A", "Big", 10.0)])
    gsp = pd.Series({"Small": 10.0, "Big": 1000.0})
    table, moved = C.allocate_components(fes, C.resolve_config({"enabled": True}), 2030, gsp,
                                         {"electric_vehicles": 100.0})
    assert table["electric_vehicles"].sum() == pytest.approx(100.0)
    assert table.loc["Small", "electric_vehicles"] <= 9.5 + 1e-9
    assert moved > 0


def test_defaults_yaml_block_matches_code_defaults_when_present():
    # The code defaults mirror CP30's demand build: the brownfield machinery (method overlay) with CP30's
    # files, editions and peak basis (source overlay), as a CP30 scenario merges them.
    from config.config_loader import deep_merge

    layers = deep_merge(load_layer("method", "brownfield"), load_layer("source", "CP30"))
    block = (layers.get("demand") or {}).get("build")
    if block is None:
        pytest.skip("demand.build not in the brownfield and CP30 overlays")
    for key in ("national_lines_file", "ed1_component_pathway"):
        assert block[key] == C.DEFAULT_CONFIG[key], key        # raw: not filled from the code defaults
    assert block["shape"]["peak_target"] == C.DEFAULT_CONFIG["shape"]["peak_target"]
    assert block["named_loads"]["file"] == C.DEFAULT_CONFIG["named_loads"]["file"]
    for key in ("FES_year", "FES_scenario"):
        assert block["spatial"][key] == C.DEFAULT_CONFIG["spatial"][key], key
    resolved = C.resolve_config(block)
    for key in ("spatial", "named_loads", "national_lines_file", "capacity_redistribution",
                "capacity_safety_margin", "min_supergrid_kv"):
        assert resolved[key] == C.DEFAULT_CONFIG[key], key
    for key, value in C.DEFAULT_CONFIG["shape"].items():
        assert resolved["shape"][key] == value, key
