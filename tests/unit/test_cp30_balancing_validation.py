"""The NESO balancing check for a CP30 2030 family (rule validate_CP30_balancing, 18 Oct 2026)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import pytest
import yaml

from scripts.CP30.balancing_validation import (
    BENCHMARK_COLUMNS,
    BENCHMARKS_FILE,
    DEFAULT_CONFIG,
    WORKBOOK,
    ZONES,
    ZONES_FILE,
    allowed_zones,
    balancing_config,
    build_balancing_validation,
    bus_zone_map,
    cost_by_group,
    delay_case_of,
    extract_benchmarks,
    final_generation,
    interconnector_energy,
    interconnector_import_signs,
    interconnector_volumes,
    load_zone_polygons,
    read_benchmarks,
    run,
    scenario_settings,
    unit_volumes,
    zone_turndown,
)
from config.config_loader import load_layer

ROOT = Path(__file__).resolve().parents[2]


# ── Benchmarks ──────────────────────────────────────────────────────────────
@pytest.mark.unit
@pytest.mark.skipif(not (ROOT / WORKBOOK).exists(), reason="CP30 workbook not present")
def test_benchmark_file_is_the_workbook_re_read():
    """Every committed benchmark row equals a fresh read of the workbook (labels checked on the way)."""
    fresh = extract_benchmarks(ROOT / WORKBOOK)
    committed = pd.read_csv(ROOT / BENCHMARKS_FILE)
    assert list(committed.columns) == BENCHMARK_COLUMNS
    assert list(fresh["metric"]) == list(committed["metric"])
    np.testing.assert_allclose(fresh["value"].to_numpy(float), committed["value"].to_numpy(float), rtol=1e-12)
    for col in ("unit", "sheet", "cell"):
        assert list(fresh[col].fillna("")) == list(committed[col].fillna(""))


@pytest.mark.unit
def test_benchmark_values_quoted_in_the_review():
    b = read_benchmarks(ROOT / BENCHMARKS_FILE)["value"]
    assert b["cp18_offer_ic_twh"] == pytest.approx(13.833, abs=1e-3)
    assert b["cp18_bid_wind_twh"] == pytest.approx(18.031, abs=1e-3)
    assert b["n2_unabated_gas_twh"] == pytest.approx(18.539)
    assert b["n2_generation_excl_storage_twh"] == pytest.approx(373.988)
    assert b["n2_unabated_gas_share"] == pytest.approx(18539 / 373988)
    assert b["n2_net_export_twh"] == pytest.approx(28.967)
    assert b["cp08_recommended_ffr_total_twh"] == pytest.approx(25.6825, abs=1e-4)
    assert b["cp08_today_ffr_total_twh"] == pytest.approx(62.556, abs=1e-3)
    assert b["cp08_recommended_ffr_scotland_twh"] == pytest.approx(13.668, abs=1e-3)
    assert b["es1_start_bid_wind_twh"] == pytest.approx(10.762, abs=1e-3)
    assert b["es1_start_offer_ic_twh"] == pytest.approx(8.569, abs=1e-3)
    assert b["neso_constraint_cost_gbpm"] == 3570.0
    assert b["neso_cost_seven_late_gbpm"] == pytest.approx(3570.0 + 6670.0)


# ── Config ──────────────────────────────────────────────────────────────────
@pytest.mark.unit
def test_code_defaults_and_unknown_keys():
    assert balancing_config(None) == DEFAULT_CONFIG
    assert balancing_config({"cost_warn_rel": 0.2})["final_warn_rel"] == DEFAULT_CONFIG["final_warn_rel"]
    with pytest.raises(ValueError):
        balancing_config({"families": ["cp30_network"]})
    with pytest.raises(ValueError):
        balancing_config({"volume_basis": "neither"})


@pytest.mark.unit
def test_defaults_block_equals_the_code_defaults():
    """validation.balancing of the CP30 source overlay must equal DEFAULT_CONFIG."""
    block = (load_layer("source", "CP30").get("validation") or {}).get("balancing")
    assert block is not None, "validation.balancing must be in config/defaults/source/CP30.yaml"
    assert block == DEFAULT_CONFIG
    assert balancing_config(block) == DEFAULT_CONFIG


@pytest.mark.unit
def test_scenario_settings_and_delay_case():
    sc = {"validation": {"balancing": {"cost_warn_rel": 0.1}},
          "market": {"balancing": {"network_delivery": {"case": "no_acceleration_outage_neso"}}},
          "network": {"transmission": {"outage_schedule": {"secured_capability": {
              "include": {"B4": [{"component": "link", "name": "X", "sign": -1.0}]}, "capability_year": 2030}}}}}
    s = scenario_settings(sc)
    assert s["network_delivery_case"] == "no_acceleration_outage_neso"
    assert s["secured_capability"] == {"include": {"B4": [{"component": "link", "name": "X", "sign": -1.0}]}}
    assert delay_case_of(s["network_delivery_case"]) == "no_acceleration"
    assert delay_case_of("outage_neso", "CP30_2030_cp30_network") is None
    assert delay_case_of(None, "CP30_2030_seven_late") == "seven_late"


# ── Measures on synthetic tables ────────────────────────────────────────────
REDISPATCH = pd.DataFrame({
    "component": ["wf1", "wf2", "ccgt", "chp", "batt", "shed", "eu", "sink", "pv"],
    "carrier": ["wind_onshore", "wind_offshore", "CCGT", "CHP", "Battery", "load_shedding", "EU_market",
                "churn_sink", "solar_pv"],
    "increase_MWh": [0.0, 5.0, 80.0, 20.0, 10.0, 7.0, 50.0, 30.0, 0.0],
    "decrease_MWh": [100.0, 50.0, 0.0, 0.0, 4.0, 0.0, 0.0, 30.0, 6.0],
})


@pytest.mark.unit
def test_unit_volumes_follow_cp18_groups():
    v = unit_volumes(REDISPATCH)
    assert v["offer_gas"] == 100.0 and v["bid_gas"] == 0.0           # CCGT + CHP
    assert v["offer_wind"] == 5.0 and v["bid_wind"] == 150.0
    assert v["offer_storage"] == 10.0 and v["bid_storage"] == 4.0
    assert v["bid_other"] == 6.0                                      # solar; slack, foreign market, sink out
    assert v["offer_other"] == 0.0
    assert v["clean_down"] == 156.0


@pytest.mark.unit
def test_interconnector_volumes_in_import_orientation():
    idx = pd.date_range("2030-01-01", periods=3, freq="h")
    links = pd.DataFrame({"bus0": ["HVDC_External_Norway", "GB1"], "bus1": ["GB1", "HVDC_External_France"]},
                         index=["IC_A", "IC_B"])
    buses = pd.DataFrame({"country": ["GB", "NO", "FR"]}, index=["GB1", "HVDC_External_Norway", "HVDC_External_France"])
    sign = interconnector_import_signs(links, buses)
    assert sign.to_dict() == {"IC_A": 1.0, "IC_B": -1.0}
    stage1 = pd.DataFrame({"IC_A": [100.0, 100.0, 100.0], "IC_B": [0.0, 0.0, 0.0]}, index=idx)
    bal = pd.DataFrame({"IC_A": [150.0, 100.0, 50.0], "IC_B": [20.0, 0.0, -10.0]}, index=idx)
    vol = interconnector_volumes(bal, stage1, sign)
    # IC_A: +50 import (offer), -50 (bid); IC_B (bus1 foreign): p0 +20 = export (bid), -10 = import (offer)
    assert vol == {"offer_ic": 60.0, "bid_ic": 70.0}
    assert interconnector_energy(bal, sign) == {"ic_import": 310.0, "ic_export": 20.0}


@pytest.mark.unit
def test_final_generation_on_n2_groups():
    energy = pd.Series({"w": 10.0, "c": 3.0, "h": 2.0, "e": 7.0, "s": 1.0, "n": 5.0})
    carriers = pd.Series({"w": "wind_offshore", "c": "CCGT", "h": "CHP", "e": "EU_market", "s": "load_shedding",
                          "n": "nuclear"})
    f = final_generation(energy, carriers)
    assert f == {"wind": 10.0, "gas_incl_chp": 5.0, "unabated_gas": 3.0, "generation": 20.0}


@pytest.mark.unit
def test_cost_by_group_drops_the_total_row():
    c = pd.DataFrame({"carrier": ["CCGT", "wind_onshore", "TOTAL"], "offer_cost": [10.0, 0.0, 10.0],
                      "bid_cost": [0.0, 5.0, 5.0], "net_cost": [10.0, 5.0, 15.0], "increase_MWh": [1.0, 0.0, 1.0],
                      "decrease_MWh": [0.0, 2.0, 2.0]})
    g = cost_by_group(c)
    assert g["net_cost"].sum() == 15.0
    assert set(g.index) == {"gas", "wind"}


@pytest.mark.unit
def test_allowed_zones_from_boundary_sides():
    north = {b: 1 for b in ("B1a", "B2", "B4", "B6", "B7a", "B8")} | {"B9": -1, "EC5": -1, "LE1": -1, "SW1": 1,
                                                                        "B13": 1}
    assert allowed_zones(north) == ["Z1"]
    assert allowed_zones({**north, "B1a": -1, "B2": -1, "B4": -1}) == ["Z4"]
    assert allowed_zones({**north, "EC5": 1}) == ["Z12"]
    south = {b: -1 for b in ("B1a", "B2", "B4", "B6", "B7a", "B8", "B9")} | {"EC5": -1, "LE1": -1, "SW1": 1,
                                                                            "B13": 1}
    assert allowed_zones(south) == ["Z9", "Z10", "Z11"]


# ── A tiny synthetic family, end to end ─────────────────────────────────────
def _write_month(tmp: Path, name: str) -> dict:
    """One month: a Scottish wind farm, an English CCGT, a Norwegian cable, a battery; three hours."""
    from pyproj import Transformer

    t = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=3, freq="h"))
    for bus, lon, lat, country in (("SCOT1", -4.2, 57.5, "GB"), ("ENG1", -1.5, 52.5, "GB"),
                                   ("HVDC_External_Norway", 5.0, 59.0, "NO")):
        x, y = t.transform(lon, lat)
        n.add("Bus", bus, x=x, y=y)
        n.buses.loc[bus, ["lon", "lat", "country"]] = [lon, lat, country]
    n.add("Line", "L1", bus0="SCOT1", bus1="ENG1", x=0.1, s_nom=1000)
    n.add("Generator", "wf", bus="SCOT1", carrier="wind_onshore", p_nom=100)
    n.add("Generator", "ccgt", bus="ENG1", carrier="CCGT", p_nom=100)
    n.add("Generator", "eu", bus="HVDC_External_Norway", carrier="EU_market", p_nom=1000)
    n.add("StorageUnit", "batt", bus="ENG1", carrier="Battery", p_nom=10)
    n.add("Link", "IC_NSL", bus0="HVDC_External_Norway", bus1="ENG1", carrier="DC", p_nom=1400, p_min_pu=-1)
    n.generators_t.p = pd.DataFrame({"wf": [60.0, 50.0, 40.0], "ccgt": [30.0, 40.0, 50.0], "eu": [0.0, 0.0, 0.0]},
                                    index=n.snapshots)
    n.links_t.p0 = pd.DataFrame({"IC_NSL": [150.0, 100.0, 50.0]}, index=n.snapshots)
    base = tmp / name
    n.export_to_netcdf(f"{base}_balancing.nc")
    REDISPATCH.iloc[[0, 2, 4]].assign(component=["wf", "ccgt", "batt"], type="generator").to_csv(
        f"{base}_redispatch_summary.csv", index=False)
    pd.DataFrame({"carrier": ["CCGT", "wind_onshore", "TOTAL"], "offer_cost": [8000.0, 0.0, 8000.0],
                  "bid_cost": [0.0, 2000.0, 2000.0], "net_cost": [8000.0, 2000.0, 10000.0],
                  "increase_MWh": [80.0, 0.0, 80.0], "decrease_MWh": [0.0, 100.0, 100.0]}).to_csv(
        f"{base}_constraint_costs.csv", index=False)
    pd.DataFrame({"IC_NSL": [100.0, 100.0, 100.0]}, index=pd.Index(n.snapshots, name="snapshot")).to_csv(
        tmp / f"{name}_s1_wholesale_links.csv")
    (tmp / f"{name}_market_summary.json").write_text(json.dumps({"accounting": {
        "neso_comparable_redispatch_cost_GBP": 1.0e7, "interconnector_retrading_cost_GBP": 4.0e6,
        "ordinary_redispatch_cost_excluding_feasibility_slack_GBP": 6.0e6, "feasibility_slack_energy_MWh": 0.0,
        "feasibility_slack_cost_GBP": 0.0}}))
    return {"name": name, "network": f"{base}_balancing.nc", "redispatch": f"{base}_redispatch_summary.csv",
            "constraint_costs": f"{base}_constraint_costs.csv",
            "stage1_links": str(tmp / f"{name}_s1_wholesale_links.csv"),
            "market_summary": str(tmp / f"{name}_market_summary.json")}


@pytest.mark.unit
def test_tiny_family_end_to_end(tmp_path):
    month = _write_month(tmp_path, "M_Jan2030_t")
    annual = tmp_path / "annual.json"
    annual.write_text(json.dumps({"aggregate": {"neso_comparable_redispatch_cost_GBP": 3.4e9}}))
    cfg = {"benchmarks_file": str(ROOT / BENCHMARKS_FILE), "zones_file": str(ROOT / ZONES_FILE)}
    out_csv, out_json = tmp_path / "v.csv", tmp_path / "v.json"
    table = run("CP30_2030_t", [month], str(annual), str(out_csv), str(out_json), cfg,
                log_path=str(tmp_path / "v.log"))
    assert not table["status"].eq("FAIL").any()
    assert set(table["status"]) <= {"PASS", "WARN", "INFO"}
    row = table.set_index("metric")
    assert row.at["offers: unabated gas incl. CHP", "model"] == pytest.approx(80.0 / 1e6)
    assert row.at["bids: wind", "model"] == pytest.approx(100.0 / 1e6)
    assert row.at["offers: interconnectors", "model"] == pytest.approx(50.0 / 1e6)
    assert row.at["bids: interconnectors", "model"] == pytest.approx(50.0 / 1e6)
    assert row.at["offers: storage (inside other)", "model"] == pytest.approx(10.0 / 1e6)
    # Cost: the annual summary's figure, PASS within 15% of 3,570.
    assert row.at["NESO-comparable redispatch cost (headline basis: cheapest_first)", "model"] == pytest.approx(3400.0)
    assert row.at["NESO-comparable redispatch cost (headline basis: cheapest_first)", "status"] == "PASS"
    assert row.at["interconnector re-trading", "model"] == pytest.approx(4.0)
    # Final: 150 MWh of wind, 120 of CCGT, net import 300.
    assert row.at["wind", "model"] == pytest.approx(150.0 / 1e6)
    assert row.at["unabated gas (CCGT + OCGT + engines)", "model"] == pytest.approx(120.0 / 1e6)
    assert row.at["net export", "model"] == pytest.approx(-300.0 / 1e6)
    assert row.at["unabated gas share of generation", "model"] == pytest.approx(120.0 / 270.0)
    # Zones: the Scottish wind farm's 100 MWh of turn-down is all Scotland's.
    assert row.at["Scotland (Z1-4) share of clean turn-down", "model"] == pytest.approx(1.0)
    summary = json.loads(out_json.read_text())
    assert summary["delay_case"] is None and summary["score"]["central_eligible"]


@pytest.mark.unit
def test_zone_map_polygon_graph_and_external():
    polys = load_zone_polygons(ROOT / ZONES_FILE)
    from pyproj import Transformer

    t = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    rows = {"ON": (-4.2, 57.5, "GB"), "OFF": (-1.0, 58.6, "GB"), "EXT": (5.0, 59.0, "NO")}
    buses = pd.DataFrame({k: {"lon": lon, "lat": lat, "country": c, "x": t.transform(lon, lat)[0],
                              "y": t.transform(lon, lat)[1]} for k, (lon, lat, c) in rows.items()}).T
    buses[["lon", "lat", "x", "y"]] = buses[["lon", "lat", "x", "y"]].astype(float)
    net = {"buses": buses,
           "lines": pd.DataFrame({"bus0": ["ON"], "bus1": ["OFF"], "s_nom": [1.0]}, index=["L"]),
           "transformers": pd.DataFrame(columns=["bus0", "bus1", "s_nom"]),
           "links": pd.DataFrame({"bus0": ["EXT"], "bus1": ["ON"], "carrier": ["DC"], "p_nom": [1.0]}, index=["IC"])}
    bz = bus_zone_map(net, polys, crossings=None)
    assert bz.at["ON", "mapping"] == "polygon" and bz.at["ON", "neso_zone"] in ZONES
    assert bz.at["OFF", "mapping"] == "graph:ON" and bz.at["OFF", "neso_zone"] == bz.at["ON", "neso_zone"]
    assert bz.at["EXT", "mapping"] == "external" and pd.isna(bz.at["EXT", "neso_zone"])
    by_zone, unzoned = zone_turndown(
        pd.DataFrame({"component": ["g1", "g2"], "carrier": ["wind_offshore", "wind_offshore"],
                      "increase_MWh": [0.0, 0.0], "decrease_MWh": [3.0, 2.0]}),
        pd.Series({"g1": "OFF", "g2": "EXT"}), bz["neso_zone"])
    assert by_zone[bz.at["ON", "neso_zone"]] == 3.0 and unzoned == 2.0


@pytest.mark.unit
def test_delay_family_is_judged_on_its_own_figure():
    bench = read_benchmarks(ROOT / BENCHMARKS_FILE)
    measures = {"volumes": {"offer_ic": 2e7, "bid_wind": 3.5e7}, "final": {
        "wind": 2.15e8, "gas_incl_chp": 4.2e7, "unabated_gas": 3e7, "generation": 3.65e8, "ic_import": 2.3e7,
        "ic_export": 4.4e7}, "neso_comparable_cost_GBP": 7.05e9, "zones": pd.Series(1e6, index=list(ZONES)),
        "accounting": {}}
    table, score = build_balancing_validation(measures, bench, None, "no_acceleration")
    central = table[table["section"].isin(["volumes", "cost", "final", "zones"])]
    assert set(central["status"]) == {"INFO"}
    delay = table[table["section"].eq("delay")].set_index("metric")
    first = delay.iloc[0]
    assert first["neso"] == 7790.0 and first["status"] == "PASS"
    assert not score["central_eligible"]
    table_nu, _ = build_balancing_validation(measures, bench, None, "no_upgrades")
    assert (table_nu["metric"] == "clean turn-down, all zones (network we have today)").any()


def test_turn_down_basis_sets_the_headline_cost_and_keeps_the_other():
    """Pro rata turn-down (20 Oct 2026): the headline basis is named, the other basis is INFO."""
    from scripts.CP30.balancing_validation import apply_turn_down_basis

    measures = {"neso_comparable_cost_GBP": 6677e6}
    summary = {"headline_basis": "pro_rata_zonal", "mode": "closest_to_neso", "headline_basis_reason": "closer",
               "neso_comparable_redispatch_cost_by_basis_GBP": {"cheapest_first": 6677e6, "pro_rata_zonal": 7650e6}}
    out = apply_turn_down_basis(dict(measures), summary)
    assert out["headline_basis"] == "pro_rata_zonal" and out["neso_comparable_cost_GBP"] == 7650e6
    assert apply_turn_down_basis(dict(measures), None)["headline_basis"] == "cheapest_first"
    with pytest.raises(ValueError):                                   # a stale summary is refused
        apply_turn_down_basis({"neso_comparable_cost_GBP": 6000e6}, summary)
