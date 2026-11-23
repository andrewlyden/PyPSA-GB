"""Item MR3(b): the Stage 1 scarcity curve lifts the price only when margin is short."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.market.scarcity_pricing import (
    build_scarcity_pricing_callback, load_scarcity_curve, model_margin_ex_post,
)

CURVE = Path("data/market/scarcity/scarcity_curve_2024.json")


def _curve_file(tmp_path):
    # Identity map (model margin = de-rated margin) so thresholds read directly.
    curve = {"reserve_duration_h": 1.0,
             "drm_from_model_margin": {"slope": 1.0, "intercept_mw": 0.0},
             "steps": [{"drm_below_mw": 300.0, "price_gbp_per_mwh": 10.0},
                       {"drm_below_mw": 100.0, "price_gbp_per_mwh": 100.0}]}
    path = tmp_path / "curve.json"
    path.write_text(json.dumps(curve))
    return path


def _network(demand):
    import pypsa
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=len(demand), freq="h"))
    n.add("Bus", "GB")
    n.add("Generator", "cheap", bus="GB", carrier="CCGT", p_nom=1000.0, marginal_cost=50.0)
    n.add("Generator", "peaker", bus="GB", carrier="OCGT", p_nom=500.0, marginal_cost=150.0)
    n.add("Generator", "shed", bus="GB", carrier="load_shedding", p_nom=5000.0, marginal_cost=6000.0)
    n.add("Load", "d", bus="GB", p_set=pd.Series(demand, index=n.snapshots))
    return n


def _solve(n, callback):
    n.optimize(solver_name="highs", extra_functionality=callback)
    return n.buses_t.marginal_price["GB"]


def test_price_rises_by_the_step_price_only_when_margin_is_short(tmp_path):
    curve = load_scarcity_curve(_curve_file(tmp_path))
    assert curve["requirement_mw"] == pytest.approx(300.0)
    assert curve["steps"][0] == (pytest.approx(200.0), 10.0)
    # Available 1,500 MW. Demand 800 leaves 700 spare (no scarcity); 1,300 leaves
    # 200 spare (first step, +10); 1,450 leaves 50 spare (second step, +100).
    n = _network([800.0, 1300.0, 1450.0])
    base = _solve(_network([800.0, 1300.0, 1450.0]), None)
    price = _solve(n, build_scarcity_pricing_callback(n, curve))
    assert base.iloc[0] == pytest.approx(50.0) and price.iloc[0] == pytest.approx(50.0)
    assert price.iloc[1] == pytest.approx(base.iloc[1] + 10.0)
    assert price.iloc[2] == pytest.approx(base.iloc[2] + 100.0)
    # Dispatch is unchanged: scarcity prices margin, it does not shed load.
    assert n.generators_t.p["shed"].abs().max() == pytest.approx(0.0, abs=1e-6)


def test_ex_post_margin_matches_the_definition(tmp_path):
    n = _network([1300.0])
    n.optimize(solver_name="highs")
    margin = model_margin_ex_post(n, {"generators_p": n.generators_t.p})
    assert margin.iloc[0] == pytest.approx(200.0)


@pytest.mark.skipif(not CURVE.exists(), reason="calibrated curve not written")
def test_calibrated_curve_is_monotone_and_ends_at_the_reserve_scarcity_price():
    curve = json.loads(CURVE.read_text())
    prices = [s["price_gbp_per_mwh"] for s in curve["steps"]]
    assert prices == sorted(prices)
    assert curve["steps"][-1]["drm_below_mw"] == 1000.0
    assert "LoLP x VoLL" in curve["steps"][-1]["basis"]
    loaded = load_scarcity_curve(CURVE)
    # Seasonal matched thresholds: summer's model margin runs higher for the
    # same de-rated margin (NESO's summer margin carries planned maintenance).
    assert set(loaded["requirement_mw"]) == {"winter", "summer"}
    assert loaded["requirement_mw"]["summer"] > loaded["requirement_mw"]["winter"] > 20000.0
    definition = loaded["margin_definition"]
    assert definition["storage_mode"] == definition["interconnector_mode"] == "derated_capacity"
    assert 0.85 < definition["generator_factors"]["CCGT"] < 0.95
    assert definition["default_generator_factor"] == 1.0


def test_seasonal_matched_thresholds_set_step_widths_per_season(tmp_path):
    curve = {"drm_from_model_margin": {"model_thresholds_mw_by_season": {"winter": [500.0, 300.0],
                                                                         "summer": [900.0, 600.0]}},
             "steps": [{"drm_below_mw": 300.0, "price_gbp_per_mwh": 10.0},
                       {"drm_below_mw": 100.0, "price_gbp_per_mwh": 100.0}]}
    path = tmp_path / "matched.json"
    path.write_text(json.dumps(curve))
    loaded = load_scarcity_curve(path)
    assert loaded["requirement_mw"] == {"winter": 500.0, "summer": 900.0}
    assert loaded["steps"][0][0] == {"winter": 200.0, "summer": 300.0}
    # January: 1,500 MW available, demand 1,100 leaves 400 (first step, +10).
    n = _network([1100.0])
    base = _solve(_network([1100.0]), None)
    price = _solve(n, build_scarcity_pricing_callback(n, loaded))
    assert price.iloc[0] == pytest.approx(base.iloc[0] + 10.0)


def test_matched_thresholds_reproduce_neso_hours_below_each_threshold():
    from scripts.market.calibrate_scarcity_curve import match_thresholds
    idx = pd.date_range("2024-01-01", periods=24 * 366, freq="h")
    rng = np.random.default_rng(0)
    margin = pd.Series(rng.uniform(0, 40000, len(idx)), index=idx)
    drm = 0.5 * margin - 3000.0                                           # any monotone relation
    out = match_thresholds(margin, drm, [15000.0, 10000.0, 5000.0], min_hours=20)
    winter_hours = idx.month.isin([10, 11, 12, 1, 2, 3])
    winter = out["model_thresholds_mw_by_season"]["winter"]
    for threshold, value in zip([15000.0, 10000.0, 5000.0], winter):
        assert (margin[winter_hours] < value).sum() == pytest.approx((drm[winter_hours] < threshold).sum(), abs=2)


def test_margin_definition_reads_neso_capacity_market_factors():
    from scripts.market.calibrate_scarcity_curve import DERATING_FILE, build_margin_definition
    if not Path(DERATING_FILE).exists():
        pytest.skip("NESO de-rating table not downloaded")
    block = build_margin_definition()
    assert block["generator_factors"]["CCGT"] == pytest.approx(0.9048)
    assert block["generator_factors"]["OCGT"] == pytest.approx(0.9297)
    assert block["generator_factors"]["coal"] > 0.0                        # from the latest auction that priced it
    assert block["storage"]["factors_by_duration_h"]["1"] == pytest.approx(0.1115)
    assert block["interconnectors"]["factors"]["IFA2"] == pytest.approx(0.57)


def _derated_curve_file(tmp_path, definition):
    curve = {"drm_from_model_margin": {"slope": 1.0, "intercept_mw": 0.0},
             "margin_definition": definition,
             "steps": [{"drm_below_mw": 300.0, "price_gbp_per_mwh": 10.0},
                       {"drm_below_mw": 100.0, "price_gbp_per_mwh": 100.0}]}
    path = tmp_path / "curve_derated.json"
    path.write_text(json.dumps(curve))
    return path


def test_conventional_plant_counts_at_its_derating_factor(tmp_path):
    # CCGT 1,000 MW at 0.9 and OCGT 500 MW at 0.8: de-rated capacity 1,300 MW.
    # Demand 1,050 leaves 250 (first step); undeclared carriers count in full.
    definition = {"generator_factors": {"CCGT": 0.9, "OCGT": 0.8}}
    curve = load_scarcity_curve(_derated_curve_file(tmp_path, definition))
    n = _network([1050.0])
    margin_before = model_margin_ex_post(n, {"generators_p": pd.DataFrame({"cheap": [1000.0], "peaker": [50.0]},
                                                                         index=n.snapshots)},
                                         curve["margin_definition"])
    assert margin_before.iloc[0] == pytest.approx(1300.0 - 1050.0)
    base = _solve(_network([1050.0]), None)
    price = _solve(n, build_scarcity_pricing_callback(n, curve))
    assert price.iloc[0] == pytest.approx(base.iloc[0] + 10.0)


def test_storage_at_derated_capacity_counts_charging_as_demand(tmp_path):
    from scripts.market.scarcity_pricing import margin_factors, margin_members, parse_margin_definition
    definition = parse_margin_definition({"storage": {"mode": "derated_capacity",
                                                      "factors_by_duration_h": {"1": 0.1, "2": 0.2, "4": 0.45}}})
    n = _network([1000.0, 1000.0])
    n.add("StorageUnit", "bat3h", bus="GB", carrier="battery", p_nom=100.0, max_hours=3.0)
    n.add("StorageUnit", "bat30m", bus="GB", carrier="battery", p_nom=100.0, max_hours=0.5)
    factors = margin_factors(n, margin_members(n), definition)
    # 3 h takes the 2 h factor; 0.5 h is below the shortest listed duration.
    assert factors["storage"]["bat3h"] == pytest.approx(0.2)
    assert factors["storage"]["bat30m"] == pytest.approx(0.0)
    gens = pd.DataFrame({"cheap": [900.0, 1000.0], "peaker": [0.0, 100.0]}, index=n.snapshots)
    # Hour 1: storage discharges 100 (counted in its de-rated 20 MW, taken off);
    # hour 2: it charges 100, served by generators, which uses the margin.
    storage = pd.DataFrame({"bat3h": [100.0, 0.0], "bat30m": [0.0, 0.0]}, index=n.snapshots)
    margin = model_margin_ex_post(n, {"generators_p": gens, "storage_dispatch": storage}, definition)
    assert margin.iloc[0] == pytest.approx(1500.0 + 20.0 - 900.0 - 100.0)
    assert margin.iloc[1] == pytest.approx(1500.0 + 20.0 - 1100.0)


def test_interconnector_at_derated_capacity_charges_exports_to_the_margin(tmp_path):
    curve = load_scarcity_curve(_derated_curve_file(tmp_path, {
        "interconnectors": {"mode": "derated_capacity", "factors": {"IFA": 0.5}, "default_factor": 0.6}}))
    import pypsa  # noqa: F401
    n = _network([1400.0])
    n.add("Bus", "FR", country="FR")
    n.add("Generator", "fr_gen", bus="FR", carrier="EU_import", p_nom=2000.0, marginal_cost=40.0)
    n.add("Load", "fr_load", bus="FR", p_set=0.0)
    n.add("Link", "IC_IFA", bus0="FR", bus1="GB", p_nom=400.0, p_min_pu=-1.0, carrier="DC")
    # De-rated capacity 1,500 + 0.5 x 400 = 1,700. Imports are cheaper than GB
    # plant, so IFA runs at 400: margin = 1,700 - 1,000 (GB) - 400 (import) = 300,
    # exactly the top threshold, so no uplift at the GB bus.
    base = _solve(n.copy(), None)
    price = _solve(n, build_scarcity_pricing_callback(n, curve))
    assert n.links_t.p0["IC_IFA"].iloc[0] == pytest.approx(400.0)
    assert price.iloc[0] == pytest.approx(base.iloc[0])
    margin = model_margin_ex_post(n, {"generators_p": n.generators_t.p, "links_p": n.links_t.p0},
                                  curve["margin_definition"])
    assert margin.iloc[0] == pytest.approx(300.0, abs=1e-3)
    # Exporting 200 MW: no gross import, and GB plant serves 1,600 MW.
    exporting = model_margin_ex_post(
        n, {"generators_p": pd.DataFrame({"cheap": [1000.0], "peaker": [500.0], "shed": [100.0]}, index=n.snapshots),
            "links_p": pd.DataFrame({"IC_IFA": [-200.0]}, index=n.snapshots)}, curve["margin_definition"])
    assert exporting.iloc[0] == pytest.approx(1700.0 - 1500.0)


def test_margin_definition_rejects_unknown_modes_and_bad_factors():
    from scripts.market.scarcity_pricing import parse_margin_definition
    with pytest.raises(ValueError):
        parse_margin_definition({"storage": {"mode": "soc"}})
    with pytest.raises(ValueError):
        parse_margin_definition({"generator_factors": {"CCGT": 1.2}})


def test_the_shallow_band_sensitivity_drops_steps_above_the_threshold(tmp_path):
    curve = {"drm_from_model_margin": {"model_thresholds_mw_by_season": {"winter": [900.0, 600.0, 300.0],
                                                                         "summer": [950.0, 650.0, 350.0]}},
             "steps": [{"drm_below_mw": 20000.0, "price_gbp_per_mwh": 7.0},
                       {"drm_below_mw": 8000.0, "price_gbp_per_mwh": 17.0},
                       {"drm_below_mw": 4000.0, "price_gbp_per_mwh": 87.0}]}
    path = tmp_path / "c.json"
    path.write_text(json.dumps(curve))
    full = load_scarcity_curve(path)
    cut = load_scarcity_curve(path, max_drm_threshold_mw=8000)
    assert full["prices"] == [7.0, 17.0, 87.0] and cut["prices"] == [17.0, 87.0]
    assert cut["requirement_mw"] == {"winter": 600.0, "summer": 650.0}
    assert cut["steps"][0][0] == {"winter": 300.0, "summer": 300.0}
    with pytest.raises(ValueError):
        load_scarcity_curve(path, max_drm_threshold_mw=1000)
