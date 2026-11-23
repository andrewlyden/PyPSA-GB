"""Item F12: the forecast bill at CP30's own 2030 price level (CP.32), an opt-in bracket.

``price_level`` absent or ``model`` settles the year exactly as before; ``cp32`` maps every
market hour onto CP.32 by its rank in the whole year (never month by month), so the system
price lands on the curve's mean, CfD top-ups fall, the interconnector rents move by the
price change on the Stage 1 flows, and the constraint line is held as solved.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import scripts.consumer_bill.project_future_bill as module
from scripts.consumer_bill import price_level_mapping as plm
from scripts.consumer_bill.project_future_bill import (
    aggregate_inputs, load_domestic_profile, price_level_of, project_components, read_economic_curtailment,
)
from tests.unit.test_future_consumer_bill import PROFILE_PATH, fixture_inputs

ROOT = Path(__file__).resolve().parents[2]
STRIKE = 100.0
EFFICIENCY = 0.97


@pytest.fixture(scope="module")
def curve():
    return plm.load_duration_curve(ROOT / plm.DEFAULT_SETTINGS["source_file"], plm.DEFAULT_SETTINGS["pathway"])


def _year_prices():
    """A 2030 year whose first half is cheap and second half dear, with a zero-price tail in each."""
    index = pd.date_range("2030-01-01", "2031-01-01", freq="h", inclusive="left")
    rng = np.random.default_rng(12)
    first = index < pd.Timestamp("2030-07-01")
    prices = np.where(first, rng.uniform(0.0, 40.0, len(index)), rng.uniform(50.0, 120.0, len(index)))
    prices[rng.random(len(index)) < 0.2] = 0.0
    return pd.Series(prices, index=index)


def _write_inputs(tmp_path, prices, cables=True):
    """Two half-year 'months': wholesale and physical networks, link schedules and market summaries."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    halves = [prices.index < pd.Timestamp("2030-07-01"), prices.index >= pd.Timestamp("2030-07-01")]
    frames, scenarios, networks, physical, summaries, links = {}, [], [], [], [], []
    for number, mask in enumerate(halves):
        snapshots = prices.index[mask]
        name = f"half{number}"
        demand = pd.Series(1.0, index=snapshots)
        frames[name] = pd.DataFrame({"demand_mw": demand, "hours": 1.0,
                                     "price_gbp_per_mwh": prices[mask].to_numpy()}, index=snapshots)
        wholesale = tmp_path / f"{name}_wholesale.nc"
        xr.Dataset({"snapshots_snapshot": ("snapshots", snapshots),
                    "links_interconnector_name": ("links_i", ["IFA", ""]),
                    "links_foreign_market_pricing": ("links_i", ["piecewise_bidirectional", ""]),
                    "links_efficiency": ("links_i", [EFFICIENCY, 1.0])},
                   coords={"links_i": ["IFA_signed", "internal"]}).to_netcdf(wholesale)
        balancing = tmp_path / f"{name}_balancing.nc"
        xr.Dataset({"snapshots_snapshot": ("snapshots", snapshots),
                    "snapshots_generators": ("snapshots", np.ones(len(snapshots))),
                    "generators_support_type": ("generators_i", ["CfD"]),
                    "generators_carrier": ("generators_i", ["wind_offshore"]),
                    "generators_bus": ("generators_i", ["GB"]),
                    "generators_cfd_round": ("generators_i", ["AR8_modelled"]),
                    "generators_p_nom": ("generators_i", [100.0]),
                    "generators_t_p": (("snapshots", "generators_t_p_i"), np.full((len(snapshots), 1), 50.0))},
                   coords={"generators_i": ["farm"], "generators_t_p_i": ["farm"]}).to_netcdf(balancing)
        schedule = tmp_path / f"{name}_wholesale_links.csv"
        flow = np.where(np.arange(len(snapshots)) % 3 == 0, -300.0, 500.0)
        pd.DataFrame({"IFA_signed": flow, "internal": 0.0}, index=snapshots).to_csv(schedule, index_label="snapshot")
        summary = {"scenario": name,
                   "accounting": {"reported_period_hours": float(len(snapshots)), "demand_MWh": float(len(snapshots)),
                                  "gross_redispatch_MWh": 0.0},
                   "revenue": {"cfd_reference_price_source": "wholesale_system_price",
                               "cfd_total_cost": 0.0, "cfd_total_dispatch_MWh": 0.0}}
        if cables:
            summary["interconnectors"] = {"by_cable": [
                {"interconnector": "IFA", "estimated_congestion_rent_GBP": 1.0e6},
                {"interconnector": "Moyle", "estimated_congestion_rent_GBP": 2.0e5}]}
        summary_path = tmp_path / f"{name}_summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        price_path = tmp_path / f"{name}_wholesale_price.csv"
        price_path.write_text("mock prices")
        scenarios.append(name)
        networks.append(wholesale)
        physical.append(balancing)
        summaries.append(summary_path)
        links.append(schedule)
    return frames, scenarios, networks, physical, summaries, links


def _settle(monkeypatch, tmp_path, prices, cables=True, **kwargs):
    module.clear_table_cache()
    frames, scenarios, networks, physical, summaries, links = _write_inputs(tmp_path, prices, cables)
    monkeypatch.setattr(module, "read_hourly_inputs", lambda network, *args: frames[Path(network).name.split("_")[0]].copy())
    prices_paths = [tmp_path / f"{name}_wholesale_price.csv" for name in scenarios]
    return aggregate_inputs(scenarios, networks, prices_paths, summaries, 2030, physical, {"future_offshore": STRIKE},
                            load_domestic_profile(PROFILE_PATH), wholesale_links=links, **kwargs)


def test_the_price_level_defaults_to_the_model_and_refuses_unknown_levels():
    assert module.DEFAULT_PRICE_LEVEL == "model"
    assert price_level_of({}) == "model"
    assert price_level_of({"price_level": "cp32"}) == "cp32"
    with pytest.raises(ValueError, match="price_level"):
        price_level_of({"price_level": "forwards"})


def test_an_absent_or_model_price_level_leaves_the_settlement_unchanged(monkeypatch, tmp_path):
    prices = _year_prices()
    hourly_absent, annual_absent = _settle(monkeypatch, tmp_path / "a", prices)
    hourly_model, annual_model = _settle(monkeypatch, tmp_path / "b", prices, price_level="model")
    pd.testing.assert_frame_equal(hourly_absent, hourly_model)
    assert "model_price_gbp_per_mwh" not in hourly_model.columns
    assert annual_absent["price_level"] == annual_model["price_level"] == "model"
    assert "price_level_mapping" not in annual_model
    # Everything but the files' own paths (and hashes of their tmp locations) is identical.
    strip = lambda annual: {k: v for k, v in annual.items() if k != "sources"}
    assert strip(annual_absent) == strip(annual_model)
    assert annual_model["interconnector_congestion_rent_by_cable_gbp"] == {"IFA": 2.0e6, "Moyle": 4.0e5}
    # The system price is the model's own (flat demand: the time-weighted mean).
    assert annual_model["wholesale_price_gbp_per_mwh"] == pytest.approx(prices.mean(), rel=1e-12)


def test_cp32_puts_the_system_price_on_the_curve_mean_and_lowers_the_cfd_top_ups(monkeypatch, tmp_path, curve):
    prices = _year_prices()
    _, model = _settle(monkeypatch, tmp_path / "m", prices)
    hourly, cp32 = _settle(monkeypatch, tmp_path / "c", prices, price_level="cp32")
    mean = plm.curve_mean(curve)
    assert mean == pytest.approx(69.89, abs=0.01)
    # Flat demand, so the demand-weighted system price is the mapped year's time average: the curve's mean.
    assert cp32["wholesale_price_gbp_per_mwh"] == pytest.approx(mean, rel=1e-9)
    assert cp32["wholesale_price_gbp_per_mwh"] > model["wholesale_price_gbp_per_mwh"]
    assert cp32["price_level_mapping"]["mapped"]["demand_weighted_gbp_per_mwh"] == pytest.approx(mean, rel=1e-9)
    assert cp32["price_level_mapping"]["model"]["time_weighted_gbp_per_mwh"] == pytest.approx(prices.mean())
    assert cp32["model_price_statistics"]["wholesale_price_gbp_per_mwh"] == pytest.approx(
        model["wholesale_price_gbp_per_mwh"], rel=1e-12)
    assert cp32["model_price_statistics"]["domestic_profile_price_gbp_per_mwh"] == pytest.approx(
        model["domestic_profile_price_gbp_per_mwh"], rel=1e-12)
    # The hourly series keeps the model price beside the mapped one.
    pd.testing.assert_series_equal(hourly.model_price_gbp_per_mwh, prices, check_names=False, check_freq=False)
    # The domestic price and the quarters' base/peak are formed on the mapped hours too.
    assert cp32["domestic_profile_price_gbp_per_mwh"] > model["domestic_profile_price_gbp_per_mwh"]
    assert sum(q["base_price_gbp_per_mwh"] for q in cp32["base_peak_prices_by_quarter"]) > sum(
        q["base_price_gbp_per_mwh"] for q in model["base_peak_prices_by_quarter"])
    # CfD top-ups: 50 MW x (strike - reference); a higher reference pays less.
    assert model["cfd_payment_gbp"] == pytest.approx(50.0 * (STRIKE - prices).sum())
    assert cp32["cfd_payment_gbp"] == pytest.approx(50.0 * (STRIKE - hourly.price_gbp_per_mwh).sum())
    assert cp32["cfd_payment_gbp"] < model["cfd_payment_gbp"]
    # Volumes are as solved.
    assert cp32["demand_volumes_mwh"] == model["demand_volumes_mwh"]
    assert cp32["cfd_priced_generation_mwh"] == model["cfd_priced_generation_mwh"]
    assert cp32["price_level"] == "cp32"
    assert "bsuos_constraints" in cp32["price_level_held_at_model_price"]
    assert {"direct_energy", "cfd", "interconnector_cap_and_floor_pass_through"} <= set(
        cp32["price_level_repriced_lines"])


def test_cp32_ranks_the_whole_year_together_never_month_by_month(monkeypatch, tmp_path, curve):
    prices = _year_prices()
    hourly, _ = _settle(monkeypatch, tmp_path, prices, price_level="cp32")
    mapped = hourly.price_gbp_per_mwh
    first = hourly.index < pd.Timestamp("2030-07-01")
    whole_year = plm.quantile_map(prices, curve, zero_band=module.CP32_ZERO_BAND_GBP_PER_MWH)
    np.testing.assert_allclose(mapped.to_numpy(), whole_year.reindex(hourly.index).to_numpy(), rtol=0, atol=1e-9)
    # The cheap half stays below the dear half (apart from the shared zero-price tie).
    priced = prices.gt(0.0)
    assert mapped[first & priced].max() <= mapped[~first & priced].min()
    # Month by month, each half would be spread over the whole curve and average its mean.
    by_half = plm.quantile_map(prices[first], curve)
    assert float(by_half.mean()) == pytest.approx(plm.curve_mean(curve), rel=1e-9)
    assert float(mapped[first].mean()) < plm.curve_mean(curve) - 20.0


def test_cp32_moves_the_interconnector_rent_by_the_price_change_on_the_stage1_flows(monkeypatch, tmp_path):
    prices = _year_prices()
    hourly, cp32 = _settle(monkeypatch, tmp_path, prices, price_level="cp32")
    flow = pd.concat([pd.Series(np.where(np.arange(m.sum()) % 3 == 0, -300.0, 500.0), index=hourly.index[m])
                      for m in (hourly.index < pd.Timestamp("2030-07-01"), hourly.index >= pd.Timestamp("2030-07-01"))])
    change = EFFICIENCY * float(((hourly.price_gbp_per_mwh - hourly.model_price_gbp_per_mwh) * flow).sum())
    assert cp32["interconnector_congestion_rent_change_by_cable_gbp"]["IFA"] == pytest.approx(change, rel=1e-12)
    assert cp32["interconnector_congestion_rent_by_cable_gbp"]["IFA"] == pytest.approx(2.0e6 + change, rel=1e-12)
    assert cp32["interconnector_congestion_rent_by_cable_model_price_gbp"] == {"IFA": 2.0e6, "Moyle": 4.0e5}
    # Moyle has no signed link on the network: its rent is held at the model price, and said so.
    assert cp32["interconnector_congestion_rent_by_cable_gbp"]["Moyle"] == 4.0e5
    assert cp32["interconnector_rent_cables_held_at_model_price"] == ["Moyle"]
    assert any("Moyle" in note for note in cp32["price_level_held_at_model_price"].values())


def test_cp32_bill_lines_move_and_the_constraint_line_is_held(monkeypatch, tmp_path):
    prices = _year_prices()
    _, settled_model = _settle(monkeypatch, tmp_path / "m", prices)
    _, settled_cp32 = _settle(monkeypatch, tmp_path / "c", prices, price_level="cp32")
    results = {}
    for name, settled in (("model", settled_model), ("cp32", settled_cp32)):
        annual, assumptions, case, balancing, recovery, ro = fixture_inputs()
        balancing["constraint_cost_gbp"] = 50.0
        annual.update({key: value for key, value in settled.items() if key in (
            "wholesale_price_gbp_per_mwh", "domestic_profile_price_gbp_per_mwh", "domestic_shaping_gbp_per_mwh",
            "cfd_payment_gbp", "price_level", "price_level_mapping", "price_level_held_at_model_price",
            "price_level_repriced_lines")})
        results[name] = project_components(annual, assumptions, case, balancing, recovery, ro)
    line = {name: rows.set_index("component_id")["annual_gbp_per_3100kwh_ex_vat"]
            for name, (rows, _) in results.items()}
    assert line["cp32"]["direct_energy"] > line["model"]["direct_energy"]
    assert line["cp32"]["cfd"] < line["model"]["cfd"]
    assert line["cp32"]["bsuos_constraints"] == line["model"]["bsuos_constraints"]
    summary = results["cp32"][1]
    assert summary["price_level"] == "cp32"
    assert summary["price_level_mapping"]["pathway"] == "further_flex_and_renewables"
    assert any(text.startswith("Price level cp32 (item F12)") for text in summary["limitations"])
    assert results["model"][1]["price_level"] == "model"
    assert not any("item F12" in text for text in results["model"][1]["limitations"])


def test_economic_curtailment_selects_on_the_model_price_and_pays_on_the_mapped_one(tmp_path):
    module.clear_table_cache()
    snapshots = pd.date_range("2030-01-01", periods=2, freq="h")
    path = tmp_path / "wholesale.nc"
    xr.Dataset({"snapshots_snapshot": ("snapshots", snapshots),
                "snapshots_generators": ("snapshots", [1.0, 1.0]),
                "generators_support_type": ("generators_i", ["CfD"]),
                "generators_cfd_round": ("generators_i", ["AR1"]),
                "generators_carrier": ("generators_i", ["wind_offshore"]),
                "generators_p_nom": ("generators_i", [100.0]),
                "generators_t_p": (("snapshots", "generators_t_p_i"), [[50.0], [50.0]]),
                "generators_t_p_max_pu": (("snapshots", "generators_t_p_max_pu_i"), [[1.0], [1.0]])},
               coords={"generators_i": ["farm"], "generators_t_p_i": ["farm"],
                       "generators_t_p_max_pu_i": ["farm"]}).to_netcdf(path)
    mapped = pd.Series([5.0, 30.0], index=snapshots)
    model = pd.Series([0.0, 40.0], index=snapshots)
    strikes = {"AR1_offshore": STRIKE}
    paid = read_economic_curtailment(path, mapped, strikes, ["AR1"], 1.0, selection_price=model)
    # Hour 1 only (the LP's own price at zero): 50 MWh curtailed x (strike - mapped GBP5).
    assert paid["hours_at_or_below_threshold"] == 1.0
    assert paid["payment_gbp"] == pytest.approx(50.0 * (STRIKE - 5.0))
    # Without a selection price the mapped series selects, and no mapped hour is at or below GBP1.
    assert read_economic_curtailment(path, mapped, strikes, ["AR1"], 1.0)["payment_gbp"] == 0.0


def test_zero_band_ties_near_zero_prices():
    """Solver noise around zero must not order the zero-price hours (12 Oct 2026)."""
    from scripts.consumer_bill import price_level_mapping as plm
    curve = pd.Series([100.0, 50.0, 0.0, 0.0], index=[0.0, 0.5, 0.75, 1.0])
    prices = pd.Series([80.0, 60.0, 1e-6, -1e-6, 2e-7, -30.0])
    noisy = plm.quantile_map(prices, curve)
    banded = plm.quantile_map(prices, curve, zero_band=1.0)
    # the three near-zero hours map to one value once banded, and differ when not
    assert banded.iloc[2] == banded.iloc[3] == banded.iloc[4]
    assert noisy.iloc[2:5].nunique() == 3
    # the order of the remaining hours is kept and the curve's mean is preserved either way
    assert banded.iloc[0] >= banded.iloc[1] >= banded.iloc[2] >= banded.iloc[5]
    assert banded.mean() == pytest.approx(plm.curve_mean(curve))
    assert noisy.mean() == pytest.approx(plm.curve_mean(curve))
    with pytest.raises(ValueError):
        plm.quantile_map(prices, curve, zero_band=-1.0)


# ── Item F24.2: the constraint line's interconnector re-trading at the price level ──

def _retrading_summary(tmp_path, name, total, parts=None):
    document = {"scenario": name, "accounting": {"interconnector_retrading_cost_GBP": total}}
    if parts is not None:
        document["interconnectors"] = dict(parts)
    path = tmp_path / f"{name}_market_summary.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _retrading_months(tmp_path):
    jan = _retrading_summary(tmp_path, "jan", 300.0e6, {"balancing_trade_cost_GBP": 500.0e6,
                                                        "day_ahead_trade_cost_GBP": 350.0e6,
                                                        "counter_trade_premium_GBP": 150.0e6})
    feb = _retrading_summary(tmp_path, "feb", -20.0e6, {"balancing_trade_cost_GBP": -100.0e6,
                                                        "day_ahead_trade_cost_GBP": -60.0e6,
                                                        "counter_trade_premium_GBP": 20.0e6})
    return [jan, feb]


def test_retrading_is_rebuilt_from_its_foreign_valued_parts_and_keeps_its_value(tmp_path):
    record = module.retrading_at_price_level(_retrading_months(tmp_path),
                                             {"interconnector_retrading_cost_gbp": 280.0e6})
    assert record["model_gbp"] == pytest.approx(280.0e6)
    # Trade value at the foreign asks and bids, and the premium: no GB price, so the level moves none of it.
    assert record["at_price_level_gbp"] == pytest.approx(280.0e6)
    assert record["change_gbp"] == pytest.approx(0.0, abs=1e-6)
    assert record["components_gbp"] == {"balancing_trade_cost_GBP": 400.0e6, "day_ahead_trade_cost_GBP": 290.0e6,
                                        "counter_trade_premium_GBP": 170.0e6}
    assert record["months_without_components"] == []


def test_retrading_with_anything_beside_its_parts_is_refused(tmp_path):
    odd = _retrading_summary(tmp_path, "mar", 100.0e6, {"balancing_trade_cost_GBP": 80.0e6,
                                                        "day_ahead_trade_cost_GBP": 0.0,
                                                        "counter_trade_premium_GBP": 10.0e6})
    with pytest.raises(ValueError, match="is not its parts"):
        module.retrading_at_price_level([odd], {})
    # The months must be the balancing-cost extraction's own.
    with pytest.raises(ValueError, match="extraction"):
        module.retrading_at_price_level(_retrading_months(tmp_path), {"interconnector_retrading_cost_gbp": 1.0e6})


def test_a_summary_without_the_parts_keeps_its_total_and_is_listed(tmp_path):
    old = _retrading_summary(tmp_path, "old", 50.0e6)
    record = module.retrading_at_price_level([old], {"interconnector_retrading_cost_gbp": 50.0e6})
    assert record["at_price_level_gbp"] == record["model_gbp"] == 50.0e6
    assert record["months_without_components"] == ["old"]


MAPPING = {"model": {"time_weighted_gbp_per_mwh": 56.0, "demand_weighted_gbp_per_mwh": 56.1},
           "mapped": {"time_weighted_gbp_per_mwh": 69.9, "demand_weighted_gbp_per_mwh": 69.8}}


def _cp32_annual():
    return {"price_level": "cp32", "price_level_mapping": MAPPING,
            "price_level_held_at_model_price": dict(module.CP32_HELD_LINES),
            "price_level_repriced_lines": dict(module.CP32_REPRICED_LINES)}


def _cp32_balancing(**extra):
    """A balancing-cost extraction run at cp32: its CfD bids already moved (item F24.2)."""
    return {"constraint_cost_gbp": 900.0e6, "interconnector_retrading_cost_gbp": 280.0e6,
            "include_interconnector_retrading": True, "price_level": "cp32",
            "cfd_bid_price_level_change_gbp": -30.0e6, "price_level_mapping_check": MAPPING, **extra}


def test_the_balancing_record_reprices_retrading_in_the_line_and_holds_the_bids(tmp_path):
    months = _retrading_months(tmp_path)
    balancing = _cp32_balancing()
    model_balancing = {key: value for key, value in balancing.items()
                       if key not in ("price_level", "cfd_bid_price_level_change_gbp", "price_level_mapping_check")}
    # At the model's price nothing is touched.
    annual = {"price_level": "model"}
    assert module.price_level_balancing_record(annual, model_balancing, months) == (annual, model_balancing)
    annual = _cp32_annual()
    out, moved = module.price_level_balancing_record(annual, balancing, months)
    # The extraction already carries the CfD bids' change; re-trading adds none.
    assert moved["constraint_cost_gbp"] == pytest.approx(900.0e6)
    assert moved["interconnector_retrading_cost_gbp"] == pytest.approx(280.0e6)
    assert moved["interconnector_retrading_model_price_gbp"] == pytest.approx(280.0e6)
    assert balancing["constraint_cost_gbp"] == 900.0e6  # the caller's dict is not mutated
    repriced = out["price_level_repriced_lines"]
    assert "bsuos_constraints (interconnector re-trading)" in repriced
    assert "GBP-30.00M" in repriced["bsuos_constraints (CfD contract-value bids)"]
    # The rest of the accepted bids and offers stay held, storage named.
    held = out["price_level_held_at_model_price"]["bsuos_constraints"]
    assert "storage" in held and "-1.3 m" in held
    assert out["interconnector_retrading_at_price_level"]["in_constraint_line"] is True
    # Re-trading reported beside the line, not in it: recorded, the line untouched.
    beside = {**balancing, "include_interconnector_retrading": False}
    out, moved = module.price_level_balancing_record(annual, beside, months)
    assert moved is beside
    assert "bsuos_constraints (interconnector re-trading)" not in out["price_level_repriced_lines"]


def test_the_bill_refuses_an_extraction_at_another_level_or_on_another_year(tmp_path):
    months = _retrading_months(tmp_path)
    plain = {"constraint_cost_gbp": 900.0e6, "interconnector_retrading_cost_gbp": 280.0e6,
             "include_interconnector_retrading": True}
    # A cp32 bill on a model-price extraction would hold the CfD bids the twin settlement moves.
    with pytest.raises(ValueError, match="price level 'cp32' but its balancing-cost extraction"):
        module.price_level_balancing_record(_cp32_annual(), plain, months)
    # And a model bill on a cp32 extraction would carry them moved.
    with pytest.raises(ValueError, match="price level 'model'"):
        module.price_level_balancing_record({"price_level": "model"}, _cp32_balancing(), months)
    other = {**MAPPING, "model": {"time_weighted_gbp_per_mwh": 50.0, "demand_weighted_gbp_per_mwh": 56.1}}
    with pytest.raises(ValueError, match="mapped another year"):
        module.price_level_balancing_record(_cp32_annual(), _cp32_balancing(price_level_mapping_check=other), months)


def test_a_central_dispatch_line_is_held_whole_at_the_price_level(tmp_path):
    balancing = {"institution": module.CENTRAL_DISPATCH_INSTITUTION, "constraint_cost_gbp": 10.0}
    annual = {"price_level": "cp32", "price_level_held_at_model_price": dict(module.CP32_HELD_LINES)}
    out, moved = module.price_level_balancing_record(annual, balancing, [])
    assert moved is balancing
    assert "central-dispatch" in out["price_level_held_at_model_price"]["bsuos_constraints"]
    assert "interconnector_retrading_at_price_level" not in out


def test_the_bill_line_takes_the_retrading_change(monkeypatch, tmp_path):
    """A re-trading change at the level (none on today's foreign-valued parts) moves the line by itself."""
    _, settled = _settle(monkeypatch, tmp_path / "c", _year_prices(), price_level="cp32")
    rows = {}
    for name, change in (("held", 0.0), ("moved", 1.0e6)):
        annual, assumptions, case, balancing, recovery, ro = fixture_inputs()
        annual.update({key: value for key, value in settled.items() if key.startswith("price_level")})
        monkeypatch.setattr(module, "retrading_at_price_level", lambda summaries, balancing, c=change: {
            "model_gbp": 2.0e6, "at_price_level_gbp": 2.0e6 + c, "change_gbp": c, "components_gbp": {},
            "months_without_components": []})
        balancing.update({"include_interconnector_retrading": True, "interconnector_retrading_cost_gbp": 2.0e6,
                          "price_level": "cp32", "cfd_bid_price_level_change_gbp": 0.0,
                          "price_level_mapping_check": settled["price_level_mapping"]})
        annual, balancing = module.price_level_balancing_record(annual, balancing, [])
        rows[name] = project_components(annual, assumptions, case, balancing, recovery, ro)[0].set_index(
            "component_id")["annual_gbp_per_3100kwh_ex_vat"]
    assert rows["moved"]["bsuos_constraints"] > rows["held"]["bsuos_constraints"]
    assert rows["moved"].drop("bsuos_constraints").equals(rows["held"].drop("bsuos_constraints"))
