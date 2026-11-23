"""market.dispatch: the one switch that turns any scenario into a central dispatch.

``central`` attaches the overlay config/defaults/dispatch/central.yaml (after method and source, before
the scenario), the loader refuses a central scenario that changes a definitional key, and the
balancing stage reads ``pricing_basis: at_cost`` as every unit at its marginal cost.
"""

import shutil

import pytest
import yaml

from config import config_loader as cl
from scripts.market.interconnector_counter_trade import resolve_foreign_uplift
from scripts.market.pricing_basis import DEFAULT_PRICING_BASIS, priced_balancing


def _get(tree, path):
    for key in path:
        tree = tree[key]
    return tree


def test_the_central_overlay_sets_the_coupled_settings():
    layer = cl.load_layer("dispatch", "central")
    assert layer["market"]["wholesale"]["copperplate_mode"] == "none"
    assert layer["market"]["wholesale"]["scarcity_pricing"]["enabled"] is False
    assert layer["market"]["balancing"]["pricing_basis"] == "at_cost"
    assert layer["support"]["bm_bids"] == {"enabled": False, "contract_value_bids": {"enabled": False}}
    assert layer["market"]["balancing"]["mc_floors"]["wind_offshore"] == 0.01
    solver = layer["solve"]["solver"]
    assert (solver["NumericFocus"], solver["BarHomogeneous"], solver["BarIterLimit"]) == (1, 1, 400)
    assert solver["accepted_conditions"] == ["optimal", "suboptimal"]
    for path in cl.CENTRAL_DISPATCH_FIXED:
        _get(layer, path)  # every fixed key has its value in the overlay
    assert cl.load_layer("dispatch", "self") == {}


def test_base_pricing_basis_equals_the_code_default():
    assert cl.load_defaults()["market"]["balancing"]["pricing_basis"] == DEFAULT_PRICING_BASIS
    assert cl.load_defaults()["market"]["dispatch"] == "self"


def test_dispatch_mode_reads_the_entry_then_the_merged_config():
    assert cl.dispatch_mode({}, {"market": {"dispatch": "self"}}) == "self"
    assert cl.dispatch_mode({"market": {"dispatch": "CENTRAL"}}, {"market": {"dispatch": "self"}}) == "central"
    assert cl.dispatch_mode({"market": {}}, {"market": {"dispatch": "central"}}) == "central"
    with pytest.raises(ValueError, match="market.dispatch"):
        cl.dispatch_mode({"market": {"dispatch": "nodal"}}, {})


def test_a_central_scenario_may_not_change_a_definitional_key():
    layer = cl.load_layer("dispatch", "central")
    merged = cl.deep_merge(cl.deep_merge(cl.load_defaults(), layer), {"market": {"dispatch": "central"}})
    cl.check_dispatch_mode(merged, layer, "ok")
    for path, value in [(("market", "wholesale", "copperplate_mode"), "collapse"),
                        (("market", "wholesale", "scarcity_pricing", "enabled"), True),
                        (("market", "balancing", "pricing_basis"), "bid_offer"),
                        (("support", "bm_bids", "contract_value_bids", "enabled"), True)]:
        bad = cl.deep_merge(merged, {})
        _get(bad, path[:-1])[path[-1]] = value
        with pytest.raises(ValueError, match=".".join(path)):
            cl.check_dispatch_mode(bad, layer, "bad")
    # A market scenario sets them freely.
    market = cl.deep_merge(cl.load_defaults(), {"market": {"wholesale": {"copperplate_mode": "collapse"}}})
    cl.check_dispatch_mode(market, layer, "market")


def test_at_cost_reads_every_carrier_at_zero_and_keeps_its_other_keys():
    balancing = {
        "pricing_basis": "at_cost", "default_offer_markup": 0.1, "default_bid_discount": 0.1,
        "carrier_overrides": {
            "CCGT": {"mode": "markup", "offer_markup": 0.85, "bid_discount": 0.1},
            "wind_offshore": {"mode": "absolute", "offer_price": 0.0, "bid_price": 90.0},
            "Battery": {"mode": "round_trip", "bid_price": 13.0, "bid_margin": 5.0, "offer_floor": 69.0},
        },
    }
    out = priced_balancing(balancing)
    assert out["default_offer_markup"] == 0.0 and out["default_bid_discount"] == 0.0
    for spec in out["carrier_overrides"].values():
        assert spec["mode"] == "markup" and spec["offer_markup"] == 0.0 and spec["bid_discount"] == 0.0
        assert "offer_price" not in spec and "bid_price" not in spec
    assert out["carrier_overrides"]["Battery"]["offer_floor"] == 69.0
    assert balancing["carrier_overrides"]["CCGT"]["offer_markup"] == 0.85  # the config is not mutated
    assert priced_balancing({"carrier_overrides": {"CCGT": {"offer_markup": 0.85}}})["carrier_overrides"] == {
        "CCGT": {"offer_markup": 0.85}}
    with pytest.raises(ValueError, match="pricing_basis"):
        priced_balancing({"pricing_basis": "nodal"})


def test_counter_trade_borrows_zero_multipliers_at_cost():
    raw = {"foreign_plant_multipliers": "gb_gas"}
    balancing = {"carrier_overrides": {"CCGT": {"mode": "markup", "offer_markup": 0.6, "bid_discount": 0.1}}}
    assert resolve_foreign_uplift(raw, balancing)["offer"] == pytest.approx(0.6)
    at_cost = resolve_foreign_uplift(raw, {**balancing, "pricing_basis": "at_cost"})
    assert at_cost["offer"] == 0.0 and at_cost["bid"] == 0.0


def _config_dir(tmp_path, scenarios):
    config_dir = tmp_path / "config"
    shutil.copytree(cl.CONFIG_DIR / "defaults", config_dir / "defaults")
    (config_dir / "config.yaml").write_text("run_scenarios: []\n", encoding="utf-8")
    (config_dir / "scenarios.yaml").write_text(yaml.safe_dump(scenarios), encoding="utf-8")
    return config_dir


_HISTORICAL = {"modelled_year": 2020, "method": "historical",
               "targets": {"source": "historical", "edition": None, "pathway": None}}


def test_the_overlay_sits_between_the_source_and_the_scenario(tmp_path):
    scenarios = {
        "Central_test": {**_HISTORICAL, "market": {"dispatch": "central"},
                         "solve": {"solver": {"NumericFocus": 3}}},
        "Market_test": dict(_HISTORICAL),
    }
    merged = cl.load_config(_config_dir(tmp_path, scenarios))["scenarios"]
    central, market = merged["Central_test"], merged["Market_test"]
    assert central["market"]["balancing"]["pricing_basis"] == "at_cost"
    assert central["market"]["wholesale"]["copperplate_mode"] == "none"
    assert central["solve"]["solver"]["NumericFocus"] == 3          # stated: the scenario wins
    assert central["solve"]["solver"]["BarIterLimit"] == 400        # unstated: the overlay's
    assert market["market"]["balancing"]["pricing_basis"] == "bid_offer"
    assert market["solve"]["solver"]["BarIterLimit"] == 200


def test_a_contradicting_scenario_is_refused_at_load(tmp_path):
    scenarios = {"Central_test": {**_HISTORICAL, "market": {
        "dispatch": "central", "wholesale": {"copperplate_mode": "collapse"}}}}
    with pytest.raises(ValueError, match="copperplate_mode"):
        cl.load_config(_config_dir(tmp_path, scenarios))
