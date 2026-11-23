"""CP30 2030 consumer demand flexibility (follow-up F5, 27 Oct 2026): scripts/demand/cp30_flexibility.py.

The flexible parts of the CP30 shape are carved from the loads and handed to storage units on
the 2050 chain's conventions: energy conserved, nothing added on top of the smart-charged EV
share, units on the buses of the demand they shift, checks see the inflexible demand.
"""
import copy
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import pytest
import yaml

from scripts.demand import cp30_demand as C
from scripts.demand import cp30_flexibility as F
from scripts.demand import demand_carriers as dc
from scripts.demand import fes_demand as f50

LOG = logging.getLogger("test_cp30_flexibility")
ROOT = Path(__file__).resolve().parents[2]
PATHWAY = "Further Flex and Renewables"


# ───────────────────────────── helpers ─────────────────────────────

def _ed1(**over):
    """CP30 ED1 flexibility rows (FFR 2030 values, GW, negative as in the workbook)."""
    values = {"Smart charging impact at peak": -4.461721, "V2G potential at peak": -0.935879,
              "V2G maximum potential": -1.872162, "Residential DSR impact at peak": -2.68656,
              "I&C DSR impact at peak": -2.03,
              "Residential peak reduction - storage heater flex": -4.108,
              "Non-residential peak reduction - storage heater flex": -0.016,
              "Residential peak reduction - hybrid heat pump flex": -0.061,
              "Non-residential peak reduction - hybrid heat pump flex": -0.018,
              "Residential peak reduction - thermal storage flex": -0.307,
              "Non-residential peak reduction - thermal storage flex": -0.125,
              "Residential peak reduction - thermal storage and district heating flex": -1.105,
              "Non-residential peak reduction - thermal storage and district heating flex": -0.401}
    values.update(over)
    rows = [{"Data item": k, "Pathway": PATHWAY, "Unit": "GW", "2030": v} for k, v in values.items()]
    rows += [{"Data item": "Residential DSR impact at peak", "Pathway": "New Dispatch", "Unit": "GW", "2030": -1.976},
             {"Data item": "I&C DSR impact at peak", "Pathway": "New Dispatch", "Unit": "GW", "2030": -1.392}]
    return pd.DataFrame(rows)


def _snapshots(days=21):
    return pd.date_range("2030-01-01", periods=24 * days, freq="h")


def _temperature(idx):
    t = 4.0 + 3.0 * np.sin(2 * np.pi * (idx.hour.to_numpy() - 9) / 24) - 0.2 * (idx.dayofyear.to_numpy() % 7)
    return pd.Series(t, index=idx, name="temperature_c")


def _prep(gsps=("A", "B")):
    comps = pd.DataFrame({"electric_vehicles": [60.0, 40.0], "heat_pumps": [30.0, 20.0],
                          "district_heat": [6.0, 4.0]}, index=list(gsps))
    return {"accounting": {"loss_factor": 1.1}, "components": comps,
            "component_gwh": {"electric_vehicles": 100.0, "heat_pumps": 50.0, "district_heat": 10.0},
            "lines": {"residential_evs_at_peak_gw": 0.010, "commercial_evs_at_peak_gw": 0.010,
                      "residential_heat_pumps_at_peak_gw": 0.01, "commercial_heat_pumps_at_peak_gw": 0.01},
            "ev_residential_share": 0.6, "ev_depot_share": 0.1,
            "gsp_gwh": pd.Series({g: 1000.0 for g in gsps})}


def _demand_cfg(ev_smart_share=0.5):
    cfg = C.resolve_config({"enabled": True})
    cfg["shape"]["ev_smart_share"] = ev_smart_share
    return cfg


def _flex_cfg(**over):
    cfg = F.resolve_config({"enabled": True})
    for k, v in over.items():
        cfg[k] = {**cfg[k], **v} if isinstance(v, dict) else v
    return cfg


def _ed1_flex():
    return F.ed1_flexibility(_ed1(), PATHWAY, 2030, PATHWAY)


def _built(**over):
    idx = _snapshots()
    return F.build_flexibility(idx, _temperature(idx), _prep(), _demand_cfg(), _flex_cfg(**over), _ed1_flex(),
                               ev_fleet=1_000_000.0)


def _network(loads_mw=None, idx=None):
    idx = _snapshots() if idx is None else idx
    n = pypsa.Network()
    n.set_snapshots(idx)
    for b in ("bA", "bB"):
        n.add("Bus", b, carrier="AC")
    loads_mw = loads_mw or {"bA": 20000.0, "bB": 15000.0}
    for b, mw in loads_mw.items():
        n.add("Load", f"load_{b}", bus=b, carrier="electricity", p_set=pd.Series(mw, index=idx))
    return n


SHARES = pd.DataFrame({"electric_vehicles": [0.6, 0.4], "heat_pumps": [0.6, 0.4], "district_heat": [0.6, 0.4],
                       "storage_heaters": [0.5, 0.5], "ic_dsr": [0.7, 0.3], "residential_dsr": [0.5, 0.5]},
                      index=pd.Index(["bA", "bB"], name="bus"))


# ───────────────────────────── code defaults ─────────────────────────────

def test_off_by_default_and_one_key_switches_it_on():
    assert F.DEFAULT_CONFIG["enabled"] is False
    assert not F.is_enabled(F.resolve_config({})) and not F.is_enabled(None)
    assert F.is_enabled({"enabled": True})


def test_shares_the_2050_chain_parameters():
    """One method: the EV, V2G, heat and DSR parameters are fes_demand's, with the 2050 study's store_days 1.0."""
    for key in ("v2g", "heat_pump_flexibility", "district_heat_flexibility"):
        assert F.DEFAULT_CONFIG[key] == f50.DEFAULT_CONFIG[key], key
    for key in ("ic_dsr", "residential_dsr", "storage_heater_flexibility"):
        same = {k: v for k, v in F.DEFAULT_CONFIG[key].items()
                if k not in ("allocation_blocks", "reference_window", "enabled")}
        assert same == {k: v for k, v in f50.DEFAULT_CONFIG[key].items() if k != "enabled"}, key
    ev = F.DEFAULT_CONFIG["electric_vehicles"]
    for k, v in ev.items():
        if k != "store_days":
            assert v == f50.DEFAULT_CONFIG["electric_vehicles"][k], k
    assert ev["store_days"] == 1.0
    # residential DSR on in both years (user decision, 27 Oct 2026): on in 2030 code, config in 2050
    assert F.DEFAULT_CONFIG["residential_dsr"]["enabled"] is True and F.DEFAULT_CONFIG["ic_dsr"]["enabled"] is True


def test_defaults_yaml_block_matches_code_defaults_when_present():
    from config.config_loader import load_layer

    block = ((load_layer("method", "brownfield").get("demand") or {}).get("build") or {}).get("flexibility")
    assert block is not None, "demand.build.flexibility is set in config/defaults/method/brownfield.yaml"
    assert block["enabled"] is True   # F5 central in the CP30 2030 study (user, 27 Oct 2026)
    resolved = F.resolve_config(block)
    for key, value in F.DEFAULT_CONFIG.items():
        if key != "enabled":
            assert resolved[key] == value, key


def test_storage_heaters_are_consumer_flexibility_and_demand():
    assert dc.STORAGE_HEATER_FLEXIBILITY in dc.CONSUMER_FLEX_CARRIERS
    assert dc.STORAGE_HEATER_FLEXIBILITY in dc.CONSUMER_OWNED_CARRIERS
    assert set(dc.CONSUMER_FLEX_CARRIERS) | {dc.DACCS} == set(dc.DEMAND_FLEX_STORAGE_CARRIERS)
    assert dc.DEMAND_LINE_OF_CARRIER[dc.STORAGE_HEATER_FLEXIBILITY] == "residual"   # inside ED1's base


# ───────────────────────────── ED1 figures ─────────────────────────────

def test_ed1_items_are_summed_positive_and_fall_back_by_item():
    flex = _ed1_flex()
    assert flex["storage_heater"]["gw"] == pytest.approx(4.124)
    assert flex["hybrid_hp"]["gw"] == pytest.approx(0.079)
    assert flex["thermal_storage_and_dh"]["gw"] == pytest.approx(1.506)
    assert flex["ic_dsr"]["gw"] + flex["residential_dsr"]["gw"] == pytest.approx(4.71656)
    nd = F.ed1_flexibility(_ed1(), "New Dispatch", 2030, PATHWAY)
    assert nd["ic_dsr"]["gw"] == pytest.approx(1.392) and nd["ic_dsr"]["pathway"] == "New Dispatch"
    assert nd["storage_heater"]["pathway"] == PATHWAY                    # New Dispatch publishes none


def test_district_heat_basis():
    flex = _ed1_flex()
    assert F.district_heat_flex_gw(flex, "combined_less_thermal_storage") == pytest.approx(1.506 - 0.432)
    assert F.district_heat_flex_gw(flex, "combined") == pytest.approx(1.506)
    with pytest.raises(ValueError):
        F.district_heat_flex_gw(flex, "other")


# ───────────────────────────── national build ─────────────────────────────

def test_ev_flexible_pool_is_the_smart_share_already_in_the_shape():
    """No double count: the carved EV reference is exactly the overnight smart share the CP30 shape placed."""
    b = _built()
    prep, dt = _prep(), 1.0
    ev_res_mwh = 100.0 * 1e3 * prep["ev_residential_share"]
    carved = sum(b["profiles"][f"{c}_reference"].sum() * dt for c in (dc.EV_SMART_CHARGING, dc.V2G))
    assert carved == pytest.approx(0.5 * ev_res_mwh)
    idx = _snapshots()
    hours = idx.hour.to_numpy()
    window = f50.gaussian_day(hours, C.DEFAULT_CONFIG["shape"]["ev_smart_window"])
    ref = b["profiles"][f"{dc.EV_SMART_CHARGING}_reference"] + b["profiles"][f"{dc.V2G}_reference"]
    assert np.allclose(ref.to_numpy() / ref.sum(), window / window.sum())


@pytest.mark.parametrize("carrier", [dc.EV_SMART_CHARGING, dc.V2G, dc.HEAT_PUMP_FLEXIBILITY,
                                     dc.DISTRICT_HEAT_FLEXIBILITY, dc.STORAGE_HEATER_FLEXIBILITY])
def test_each_unit_drains_exactly_the_energy_carved_for_it(carrier):
    b = _built()
    spec, prof = b["flex"][carrier], b["profiles"]
    carved = prof[spec["reference"]].sum()
    electricity = -prof[spec["inflow"]].sum() / spec["efficiency_store"]
    assert electricity == pytest.approx(carved, rel=1e-9)
    assert spec["electricity_gwh"] * 1e3 == pytest.approx(carved, rel=1e-9)


def test_v2g_discharge_at_peak_and_maximum_follow_ed1():
    b = _built()
    spec, prof = b["flex"][dc.V2G], b["profiles"]
    assert spec["discharge_mw"] == pytest.approx(1872.162)
    peak_pu = prof["V2G_fleet_p_max_pu"].max() * spec["p_nom_mw"]
    assert peak_pu == pytest.approx(1872.162, rel=1e-3)               # whole fleet plugged in
    assert spec["vehicles"] == pytest.approx(min(0.5 * 1_000_000, 1872.162e3 / 7.0))


def test_heat_shares_follow_ed1_at_peak_and_cap_at_the_whole_load():
    b = _built()
    cal = b["calibration"]
    assert cal["heat_pumps_flex_share"] == pytest.approx(min(1.0, (0.079 + 0.432) * 1.1e3 / cal["heat_pumps_at_peak_mw"]))
    assert cal["district_heat_flex_share"] <= 1.0
    spec = b["flex"][dc.HEAT_PUMP_FLEXIBILITY]
    drain = -b["profiles"][spec["inflow"]]
    assert spec["p_nom_mw"] == pytest.approx(1.5 * drain.max())


def test_storage_heaters_charge_on_economy7_and_hold_the_coldest_day():
    b = _built()
    spec, prof = b["flex"][dc.STORAGE_HEATER_FLEXIBILITY], b["profiles"]
    ref, drain = prof[spec["reference"]], -prof[spec["inflow"]]
    day = ref.index.normalize()
    assert np.allclose(ref.groupby(day).sum(), drain.groupby(day).sum())   # each day's energy, same day
    assert ref[ref.index.hour.isin([12, 15, 18])].max() < 0.02 * ref.max()  # nothing carved in the day
    coldest = drain.groupby(day).sum().max()
    assert spec["p_nom_mw"] == pytest.approx(coldest / 7.0) and spec["energy_mwh"] == pytest.approx(coldest)
    # the direct-acting load at the ACS-peak hours is ED1's figure (x loss factor)
    assert b["calibration"]["storage_heater_direct_at_peak_mw"] == pytest.approx(4.124 * 1.1e3)


def test_demand_response_is_energy_neutral_with_residential_on():
    b = _built()
    spec = b["flex"][dc.IC_DSR]
    assert spec["p_nom_mw"] == pytest.approx(2030.0) and spec["energy_mwh"] == pytest.approx(4060.0)
    assert spec["inflow"] is None and spec["reference"] is None and spec["electricity_gwh"] == 0.0
    res = b["flex"][dc.RESIDENTIAL_DSR]
    assert res["p_nom_mw"] == pytest.approx(2686.56) and res["energy_mwh"] == pytest.approx(2 * 2686.56)
    assert res["marginal_cost"] == 30.0 and res["allocation"] == "residential_dsr"
    nd = F.build_flexibility(_snapshots(), _temperature(_snapshots()), _prep(), _demand_cfg(), _flex_cfg(),
                             F.ed1_flexibility(_ed1(), "New Dispatch", 2030, PATHWAY), ev_fleet=1_000_000.0)
    assert nd["flex"][dc.RESIDENTIAL_DSR]["p_nom_mw"] == pytest.approx(1976.0)        # New Dispatch's own row
    off = _built(residential_dsr={"enabled": False})
    assert dc.RESIDENTIAL_DSR not in off["flex"]


def test_switching_a_block_off_drops_its_units():
    b = _built(storage_heater_flexibility={"enabled": False}, v2g={"enabled": False})
    assert dc.STORAGE_HEATER_FLEXIBILITY not in b["flex"] and dc.V2G not in b["flex"]
    ev = b["flex"][dc.EV_SMART_CHARGING]
    assert ev["electricity_gwh"] == pytest.approx(0.5 * 0.6 * 100.0)   # the whole pool, smart charging only


# ───────────────────────────── network step ─────────────────────────────

def test_integration_conserves_demand_and_places_units_on_the_component_buses():
    n = _network()
    b = _built()
    before = n.loads_t.p_set.copy()
    stats = F.integrate_flexibility(n, b, SHARES, LOG)
    assert stats["loads_after_gwh"] + stats["drain_electricity_gwh"] == pytest.approx(stats["loads_before_gwh"])
    assert stats["kept_as_fixed_gwh"] == pytest.approx(0.0)
    assert (n.loads_t.p_set.to_numpy() >= -1e-9).all()
    su = n.storage_units
    ev = su[su.carrier.eq(dc.EV_SMART_CHARGING)]
    assert set(ev.bus) == {"bA", "bB"}
    assert ev.loc["bA EV_smart_charging", "p_nom"] / ev["p_nom"].sum() == pytest.approx(0.6)
    assert (su["state_of_charge_initial"] == 0).all() and not su["cyclic_state_of_charge"].any()
    # per bus: removed energy = the units' drain electricity there
    removed = (before - n.loads_t.p_set).sum()
    for bus in ("bA", "bB"):
        units = su[su.bus.eq(bus) & su.carrier.isin(F.CARVED_CARRIERS)]
        drained = sum(-n.storage_units_t.inflow[u].sum() / units.at[u, "efficiency_store"] for u in units.index)
        assert removed[f"load_{bus}"] == pytest.approx(drained, rel=1e-9)


def test_a_bus_that_cannot_give_its_reference_keeps_part_as_fixed_load():
    n = _network({"bA": 20000.0, "bB": 0.5})                  # bB's load is too small for its share
    stats = F.integrate_flexibility(n, _built(), SHARES, LOG)
    assert stats["buses_short"] == 1 and stats["kept_as_fixed_gwh"] > 0
    assert (n.loads_t.p_set["load_bB"] >= -1e-9).all()
    assert stats["loads_after_gwh"] + stats["drain_electricity_gwh"] == pytest.approx(stats["loads_before_gwh"])


def test_reference_restores_the_as_built_loads_exactly():
    n = _network()
    before = n.loads_t.p_set.copy()
    b = _built()
    F.integrate_flexibility(n, b, SHARES, LOG)
    meta = {F.META_KEY: {"reference": F.reference_record(n.snapshots, b)}}
    restored = F.restore_reference_loads(n.loads, n.loads_t.p_set, n.storage_units, meta, n.snapshots)
    assert np.allclose(restored.to_numpy(), before.to_numpy(), atol=1e-2)
    # a solved week is aligned by time
    week = n.snapshots[48:216]
    part = F.restore_reference_loads(n.loads, n.loads_t.p_set.loc[week], n.storage_units, meta, week)
    assert np.allclose(part.to_numpy(), before.loc[week].to_numpy(), atol=1e-2)
    # no record: unchanged
    same = F.restore_reference_loads(n.loads, n.loads_t.p_set, n.storage_units, {}, n.snapshots)
    assert same.equals(n.loads_t.p_set)


def test_locational_tables_see_storage_plant_and_the_as_built_demand():
    from scripts.CP30.locational_validation import NetworkTables

    n = _network()
    before = n.loads_t.p_set.copy()
    n.add("StorageUnit", "bat", bus="bA", carrier="Battery", p_nom=50.0, max_hours=2.0)
    b = _built()
    F.integrate_flexibility(n, b, SHARES, LOG)
    n.meta = {F.META_KEY: {"reference": F.reference_record(n.snapshots, b)}}
    t = NetworkTables.from_network(n)
    assert list(t.storage_units.index) == ["bat"]
    assert np.allclose(t.load_p.to_numpy(), before.to_numpy(), atol=1e-2)


def test_market_churn_sink_and_share_ignore_consumer_flexibility():
    from scripts.market.market_utils import apply_churn_sink, apply_storage_churn_share

    n = _network(idx=pd.date_range("2030-01-01", periods=3, freq="h"))
    n.add("StorageUnit", "bat", bus="bA", carrier="Battery", p_nom=100.0, max_hours=2.0, marginal_cost=0.0)
    n.add("StorageUnit", "bA EV_smart_charging", bus="bA", carrier=dc.EV_SMART_CHARGING, p_nom=1000.0, max_hours=1.0)
    n.add("StorageUnit", "bB IC_DSR", bus="bB", carrier=dc.IC_DSR, p_nom=500.0, max_hours=2.0, marginal_cost=0.0)
    cfg = {"churn_sink": {"enabled": True, "share_of_storage_power": 0.5},
           "renewable_curtailment_order": {"ar4_plus_bid_gbp_per_mwh": -10.0}, "storage_churn_share": 0.0}
    assert apply_churn_sink(n, cfg, LOG) == 1
    assert n.generators.loc["churn_sink bA", "p_nom"] == pytest.approx(50.0)
    apply_storage_churn_share(n, cfg, LOG)
    assert n.storage_units.at["bB IC_DSR", "marginal_cost"] == 0.0
    assert n.storage_units.at["bat", "marginal_cost"] > 0.0


def test_flexibility_units_aggregate_exactly_per_carrier():
    """The copperplate collapse merges units per carrier: every unit of a carrier has one per-unit shape."""
    n = _network()
    F.integrate_flexibility(n, _built(), SHARES, LOG)
    su = n.storage_units
    for carrier in F.CARVED_CARRIERS:
        units = su.index[su.carrier.eq(carrier)]
        assert su.loc[units, "max_hours"].nunique() == 1
        per_unit = n.storage_units_t.inflow[units].div(su.loc[units, "p_nom"], axis=1)
        assert np.allclose(per_unit.to_numpy(), per_unit.iloc[:, [0]].to_numpy())
