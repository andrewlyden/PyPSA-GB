"""Item N1: Stage 1 can dispatch on resource cost instead of market-shaped marginal cost."""

import logging

import numpy as np
import pandas as pd
import pytest

from scripts.market.marginal_cost_basis import (
    DEFAULT_RENEWABLE_CARRIERS, apply_marginal_cost_basis, needs_strike_table, resource_cost_table, settings,
)
from config.config_loader import load_defaults


def _generators():
    return pd.DataFrame({
        "carrier": ["wind_offshore", "wind_onshore", "solar_pv", "CCGT", "biomass", "nuclear", "load_shedding", "large_hydro"],
        "marginal_cost": [-156.0, 0.0, -10.0, 119.35, 60.0, 10.0, 6000.0, 0.0],
        "p_nom": [1000.0, 500.0, 200.0, 800.0, 300.0, 1200.0, 50.0, 100.0],
    }, index=["roc_wind", "cfd_wind", "emp_solar", "gas", "bio_roc", "szb", "shed", "hydro"])


def _audit():
    return pd.DataFrame({
        "generator": ["roc_wind", "cfd_wind", "emp_solar", "gas", "bio_roc", "szb", "shed"],
        "source": ["subsidy_ROC", "subsidy_CfD", "renewable_empirical", "formula+vom", "formula_ROC(1.5x73)", "nuclear_override", "protected"],
        "fuel_cost_gbp_per_mwh": [0.0, 0.0, 0.0, 107.0, 120.0, 0.0, 0.0],
        "carbon_cost_gbp_per_mwh": [0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0],
        "variable_om_gbp_per_mwh": [0.0, 0.0, 0.0, 3.35, 8.0, 0.0, 0.0],
    })


#: The plain floor, without the support order (on by default).
FLAT = {"support_order": {"enabled": False}}


class _Net:
    def __init__(self):
        self.generators = _generators()
        self.generators_t = type("T", (), {})()
        self.generators_t.marginal_cost = pd.DataFrame({"roc_wind": [-156.0, -156.0]})
        self.meta = {}


def test_settings_validate():
    assert settings(None)["mode"] == "as_built"
    assert settings({"mode": "resource_cost"})["renewable_carriers"] == DEFAULT_RENEWABLE_CARRIERS
    with pytest.raises(ValueError, match="mode"):
        settings({"mode": "cheapest"})
    with pytest.raises(ValueError, match="renewable_floor"):
        settings({"mode": "resource_cost", "renewable_floor": 0.0})


def test_table_floors_renewables_and_strips_subsidy_layers_from_the_audit():
    cfg = settings({"mode": "resource_cost", "renewable_floor": 0.01, **FLAT})
    table = resource_cost_table(_generators(), cfg, _audit())
    assert table.loc[["roc_wind", "cfd_wind", "emp_solar", "hydro"], "resource_cost"].eq(0.01).all()
    assert table.loc[["roc_wind", "cfd_wind", "emp_solar", "hydro"], "rule"].eq("renewable_floor").all()
    # Biomass with a ROC deduction goes back to fuel + carbon + VOM from the audit.
    assert table.loc["bio_roc", "resource_cost"] == pytest.approx(128.0)
    assert table.loc["bio_roc", "rule"] == "audit_fuel_carbon_vom"
    # Formula thermal, nuclear override and load shedding are never touched.
    assert table.loc[["gas", "szb", "shed"], "rule"].eq("unchanged").all()
    assert table.loc["gas", "resource_cost"] == 119.35 and table.loc["shed", "resource_cost"] == 6000.0


def test_without_an_audit_only_the_renewable_floor_applies():
    cfg = settings({"mode": "resource_cost", **FLAT})
    table = resource_cost_table(_generators(), cfg, None)
    assert table.loc["bio_roc", "rule"] == "unchanged" and table.loc["bio_roc", "resource_cost"] == 60.0
    assert table.loc["roc_wind", "resource_cost"] == 0.01


def test_apply_changes_the_network_and_reports():
    net = _Net()
    summary = apply_marginal_cost_basis(net, {"mode": "resource_cost", **FLAT}, logging.getLogger("t"), audit=_audit())
    assert summary["mode"] == "resource_cost" and summary["audit_available"]
    assert net.generators.loc["roc_wind", "marginal_cost"] == 0.01
    assert net.generators.loc["bio_roc", "marginal_cost"] == pytest.approx(128.0)
    assert net.generators.loc["gas", "marginal_cost"] == 119.35
    assert summary["negative_costs_before"] == 2 and summary["negative_costs_after"] == 0
    # cfd_wind was 0.0 -> 0.01 counts as re-costed; szb, gas, shed do not.
    assert summary["generators_recosted"] == 5
    assert summary["capacity_recosted_mw"] == pytest.approx(1000 + 500 + 200 + 300 + 100)
    # A time-varying cost on a re-costed generator would silently win, so it is dropped.
    assert summary["time_varying_costs_dropped"] == 1 and "roc_wind" not in net.generators_t.marginal_cost.columns
    assert net.meta["marginal_cost_basis"]["mode"] == "resource_cost"


def test_as_built_is_a_no_op():
    net = _Net()
    before = net.generators["marginal_cost"].copy()
    assert apply_marginal_cost_basis(net, {"mode": "as_built"}, audit=_audit()) == {"mode": "as_built"}
    pd.testing.assert_series_equal(net.generators["marginal_cost"], before)


def test_repository_default_is_as_built():
    import yaml
    from pathlib import Path
    block = load_defaults()["market"]["wholesale"]["marginal_cost_basis"]
    assert settings(block)["mode"] == "as_built"


class _Strikes:
    """Test stub: strikes by (carrier, round)."""
    STRIKES = {("wind_offshore", "AR6"): 80.0, ("wind_offshore", "FIDeR"): 201.0, ("solar_pv", "AR4"): 60.0}

    def lookup(self, carrier, cfd_round):
        strike = self.STRIKES.get((carrier, cfd_round))
        return strike, f"{carrier}|{cfd_round}", None


def _supported():
    return pd.DataFrame({
        "carrier": ["wind_offshore", "wind_offshore", "wind_onshore", "solar_pv", "wind_onshore", "CCGT"],
        "marginal_cost": [0.0, 0.0, -51.3, 0.0, 0.0, 119.35],
        "support_type": ["CfD", "CfD", "ROC", "CfD", "merchant", ""],
        "cfd_round": ["AR6", "FIDeR", "", "AR4", "", ""],
        "p_nom": [1000.0, 500.0, 200.0, 300.0, 400.0, 800.0],
    }, index=["ar6", "fider", "roc", "ar4_solar", "merchant", "gas"])


def test_support_order_turns_merchant_down_first_and_the_highest_support_last():
    # The 14 Oct 2026 decision (Ofgem 2023 A1.18): subsidised renewables give way after merchant.
    cfg = settings({"mode": "resource_cost"})
    assert cfg["support_order"] == {"enabled": True, "spread_gbp_per_mwh": 1.0, "labels_file": None}
    table = resource_cost_table(_supported(), cfg, None, _Strikes())
    cost = table["resource_cost"]
    assert cost["merchant"] == pytest.approx(1.01)
    assert cost["fider"] == pytest.approx(0.01)
    assert cost["ar6"] == pytest.approx(0.01 + 1.0 * (1 - 80.0 / 201.0))
    assert cost["roc"] == pytest.approx(0.01 + 1.0 * (1 - 51.3 / 201.0))
    assert cost["merchant"] > cost["roc"] > cost["ar4_solar"] > cost["ar6"] > cost["fider"]
    assert table.loc["gas", "rule"] == "unchanged"
    assert table.loc[["ar6", "merchant"], "rule"].eq("renewable_floor_support_order").all()
    # Every renewable stays within the floor's band: the spread only resolves the tie.
    assert cost.drop("gas").between(0.01, 1.01).all()


def test_support_order_needs_the_strike_table_for_cfd_units():
    assert needs_strike_table({"mode": "resource_cost"})
    assert not needs_strike_table({"mode": "as_built"})
    assert not needs_strike_table({"mode": "resource_cost", **FLAT})
    with pytest.raises(ValueError, match="strike table"):
        resource_cost_table(_supported(), settings({"mode": "resource_cost"}), None, None)
    with pytest.raises(ValueError, match="unknown keys"):
        settings({"mode": "resource_cost", "support_order": {"spread": 1.0}})
    with pytest.raises(ValueError, match="spread_gbp_per_mwh"):
        settings({"mode": "resource_cost", "support_order": {"spread_gbp_per_mwh": 0.0}})


def test_a_network_without_labels_orders_on_the_support_labels_file():
    # 11 Nov 2026: a FES long-range network carries no support labels; the labels file gives each generator's
    # weighted live CfD strike, so the CfD-heavy unit gives way last and the merchant one first.
    from scripts.market.marginal_cost_basis import labels_support_value

    gens = pd.DataFrame({"carrier": ["wind_offshore", "wind_offshore", "wind_onshore"],
                         "marginal_cost": [0.0, 0.0, 0.0], "p_nom": [100.0, 100.0, 100.0]},
                        index=["a", "b", "c"])
    labels = pd.DataFrame({
        "component": ["a", "a", "b", "c"], "component_type": "Generator",
        "support_type": ["cfd", "merchant", "cfd", "merchant"], "weight": [0.5, 0.5, 1.0, 1.0],
        "live_share": [1.0, None, 0.5, None], "strike_gbp_per_mwh_2024": [80.0, None, 60.0, None]})
    value = labels_support_value(gens.index, labels)
    assert value.to_dict() == {"a": 40.0, "b": 30.0, "c": 0.0}
    cfg = settings({"mode": "resource_cost", "support_order": {"labels_file": "x.csv"}})
    table = resource_cost_table(gens, cfg, None, None, labels)
    cost = table["resource_cost"]
    assert cost["c"] > cost["b"] > cost["a"]
