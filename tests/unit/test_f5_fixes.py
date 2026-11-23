"""F5 consumer flexibility fixes (28 Oct 2026): the BM arbitrage check, DSR holding cost, storage-heater
charge windows, drained-unit holding cost, the period-end cap, the shedding cap, the BM state of charge,
the validation demand line and the small-unit drop."""

import logging

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.demand import demand_carriers as dc
from scripts.demand import fes_demand as f50
from scripts.demand import cp30_flexibility as F

LOG = logging.getLogger("test_f5_fixes")


# ───────────── 1. BM cross-carrier arbitrage check leaves consumer flexibility out ─────────────

def _storage(carriers):
    idx = [f"u{i}_{c}" for i, c in enumerate(carriers)]
    return pd.DataFrame({"carrier": carriers, "efficiency_store": 0.9, "efficiency_dispatch": 0.9,
                         "marginal_cost": 0.0, "p_nom": 10.0, "max_hours": 2.0, "bus": "b"}, index=idx)


def test_arbitrage_check_ignores_held_consumer_flexibility():
    from scripts.market.market_utils import _check_storage_cross_carrier_arbitrage

    su = _storage(["Battery", "DH_flexibility", "SH_flexibility"])
    offers = pd.Series([90.0, 0.0, 0.0], index=su.index)
    bids = pd.Series([-40.0, 0.0, 0.0], index=su.index)
    _check_storage_cross_carrier_arbitrage(su, offers, bids)     # flex 0 offers do not count
    plant = _storage(["Battery", "LAES"])
    with pytest.raises(ValueError, match="cross-unit arbitrage"):
        _check_storage_cross_carrier_arbitrage(plant, pd.Series([90.0, 0.0], index=plant.index),
                                               pd.Series([-40.0, 0.0], index=plant.index))


def test_2050_overrides_without_a_storage_heater_row_pass_the_check():
    """The 2050 study's carrier_overrides have no SH_flexibility row (its derived offer is about 0)."""
    from config.config_loader import get_scenario
    from scripts.market.market_utils import _derive_storage_prices

    sc = get_scenario("FES25_HT_2050_Jan_market")
    bal = sc["market"]["balancing"]
    overrides = bal.get("carrier_overrides", {})
    assert "SH_flexibility" not in overrides
    su = _storage(["Battery", "LAES", "SH_flexibility", "residential_DSR", "EV_smart_charging"])
    su["marginal_cost"] = [1.0, 1.0, 0.0, 30.0, 0.0]
    offer, bid = _derive_storage_prices(su, overrides, float(bal.get("default_offer_markup", 0.1)),
                                        float(bal.get("default_bid_discount", 0.1)), LOG)
    assert offer.notna().all() and bid.notna().all()


# ───────────── 2. DSR holding cost and storage-heater charge windows (config, both years) ─────────────

def test_defaults_carry_the_new_keys_and_cp30_takes_the_2050_values():
    d = f50.DEFAULT_CONFIG
    assert d["ic_dsr"]["holding_cost_gbp_per_mwh_h"] == 15.0 == d["residential_dsr"]["holding_cost_gbp_per_mwh_h"]
    assert d["ic_dsr"]["holding_cost_gbp_per_mwh_h"] == d["ic_dsr"]["marginal_cost_gbp_per_mwh"] / d["ic_dsr"]["shift_hours"]
    assert d["storage_heater_flexibility"]["charge_windows"] == [[0, 7], [13, 16]]
    assert d["drained_holding_cost_gbp_per_mwh_h"] == 0.005
    c = F.DEFAULT_CONFIG
    assert c["storage_heater_flexibility"]["charge_windows"] == d["storage_heater_flexibility"]["charge_windows"]
    assert c["ic_dsr"]["holding_cost_gbp_per_mwh_h"] == 15.0
    assert c["drained_holding_cost_gbp_per_mwh_h"] == d["drained_holding_cost_gbp_per_mwh_h"]


def test_config_files_state_the_code_defaults():
    from config.config_loader import load_layer

    block = load_layer("method", "brownfield")["demand"]["build"]["flexibility"]
    resolved = F.resolve_config(block)
    for key in ("drained_holding_cost_gbp_per_mwh_h",):
        assert block[key] == F.DEFAULT_CONFIG[key]
    assert block["storage_heater_flexibility"]["charge_windows"] == F.DEFAULT_CONFIG["storage_heater_flexibility"]["charge_windows"]
    assert resolved["ic_dsr"] == F.DEFAULT_CONFIG["ic_dsr"]


def test_charge_window_profile_and_refusal():
    idx = pd.date_range("2030-01-01", periods=48, freq="h")
    pu = f50.charge_window_pu(idx, [[0, 7], [13, 16]], 7.0)
    hours = idx.hour
    assert (pu[(hours < 7) | ((hours >= 13) & (hours < 16))] == -1.0).all()
    assert (pu[(hours >= 7) & (hours < 13)] == 0.0).all() and (pu[hours >= 16] == 0.0).all()
    with pytest.raises(ValueError, match="fewer than charge_hours"):
        f50.charge_window_pu(idx, [[0, 5]], 7.0)


def test_storage_heater_unit_carries_its_window_and_stays_feasible():
    idx = pd.date_range("2030-01-01", periods=24 * 14, freq="h")
    temp = pd.Series(5.0 + 5.0 * np.sin(np.arange(len(idx)) / 24.0), index=idx)
    mask = np.zeros(len(idx), bool)
    mask[17::24] = True
    block = f50.resolve_config({})["storage_heater_flexibility"]
    unit = f50.storage_heater_flexibility(temp, mask, 1000.0, block, f50.resolve_config({}))
    assert unit["spec"]["p_min_pu"] == "SH_flexibility_p_min_pu"
    open_hours = (unit["p_min_pu"] < 0).reshape(-1, 24).sum(axis=1)
    # Every day's energy can be charged inside its windows at p_nom (coldest day / charge_hours).
    assert (open_hours * unit["spec"]["p_nom_mw"] >= unit["daily_mwh"][::24] - 1e-6).all()


def test_holding_costs_are_set_by_kind():
    flex = {dc.IC_DSR: {"inflow": None}, dc.RESIDENTIAL_DSR: {"inflow": None},
            dc.EV_SMART_CHARGING: {"inflow": "x"}, dc.STORAGE_HEATER_FLEXIBILITY: {"inflow": "y"}}
    f50.holding_costs(flex, f50.resolve_config({}))
    assert flex[dc.IC_DSR]["marginal_cost_storage"] == 15.0
    assert flex[dc.RESIDENTIAL_DSR]["marginal_cost_storage"] == 15.0
    assert flex[dc.EV_SMART_CHARGING]["marginal_cost_storage"] == 0.005
    assert flex[dc.STORAGE_HEATER_FLEXIBILITY]["marginal_cost_storage"] == 0.005


def _flex_network(n_hours=24, price=0.0, drain=10.0, with_dsr=False, holding=0.0):
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-10-20", periods=n_hours, freq="h"))
    n.add("Bus", "b")
    n.add("Generator", "cheap", bus="b", p_nom=1000.0, marginal_cost=price)
    n.add("Load", "load", bus="b", p_set=50.0)
    n.add("StorageUnit", "b SH_flexibility", bus="b", carrier=dc.STORAGE_HEATER_FLEXIBILITY, p_nom=100.0,
          max_hours=10.0, p_min_pu=-1.0, p_max_pu=0.0, cyclic_state_of_charge=False, state_of_charge_initial=0.0,
          inflow=pd.Series(-drain, index=n.snapshots), marginal_cost_storage=holding)
    return n


def test_drained_holding_cost_stops_free_surplus_charging():
    """At a zero price an unpriced store is indifferent; a holding cost makes it charge just in time."""
    n = _flex_network(holding=0.005)
    n.optimize(solver_name="highs")
    soc = n.storage_units_t.state_of_charge["b SH_flexibility"]
    assert soc.max() == pytest.approx(0.0, abs=1e-6)
    assert n.storage_units_t.p_store["b SH_flexibility"].sum() == pytest.approx(10.0 * 24, rel=1e-9)


def test_dsr_holding_cost_limits_how_long_energy_is_held():
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=12, freq="h"))
    n.add("Bus", "b")
    prices = [10.0] + [50.0] * 10 + [200.0]   # cheapest 11 h ahead, nearly as cheap 1 h ahead
    n.add("Generator", "g", bus="b", p_nom=1000.0, marginal_cost=pd.Series(prices, index=n.snapshots))
    n.add("Load", "load", bus="b", p_set=50.0)
    for name, hold in (("free", 0.0), ("held", 15.0)):
        m = n.copy()
        m.add("StorageUnit", "dsr", bus="b", carrier=dc.IC_DSR, p_nom=10.0, max_hours=2.0, p_min_pu=-1.0,
              p_max_pu=1.0, marginal_cost=30.0, marginal_cost_storage=hold, cyclic_state_of_charge=False,
              state_of_charge_initial=0.0)
        m.optimize(solver_name="highs")
        store = m.storage_units_t.p_store["dsr"]
        assert float(m.storage_units_t.p_dispatch["dsr"].iloc[-1]) > 1.0     # the shift to the dear hour pays
        if name == "free":
            assert float(store.iloc[0]) > 1.0          # charged 11 h ahead, held all that time
        else:
            assert float(store.iloc[0]) == pytest.approx(0.0, abs=1e-6)   # 11 h x 15 > the 40 saved
            assert float(store.iloc[-2]) > 1.0         # charged the hour before


# ───────────── 3. Soft period-end cap on consumer flexibility ─────────────

def test_month_end_cap_prices_flex_energy_left_above_its_start():
    from scripts.market.market_utils import build_month_end_soc_callback

    n = _flex_network(price=-1.0)          # paid to charge: without a cap it fills the store
    cb = build_month_end_soc_callback(n, {"storage_soc": {"month_end": "at_least_initial"}}, n.snapshots[-1], LOG)
    n.optimize(solver_name="highs", extra_functionality=cb)
    assert n.storage_units_t.state_of_charge["b SH_flexibility"].iloc[-1] == pytest.approx(0.0, abs=1e-6)


# ───────────── 4. Load-shedding cap includes the held flex draw ─────────────

def test_shedding_cap_adds_extra_local_load():
    from scripts.solve.solve_network import apply_load_shedding_limits

    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=3, freq="h"))
    n.add("Bus", ["a", "b"])
    n.add("Load", "la", bus="a", p_set=pd.Series([10.0, 10.0, 10.0], index=n.snapshots))
    n.add("Generator", ["ls_a", "ls_b"], bus=["a", "b"], carrier="load_shedding", p_nom=100.0)
    extra = pd.DataFrame({"a": [5.0, 0.0, 0.0], "b": [7.0, 7.0, 0.0]}, index=n.snapshots)
    apply_load_shedding_limits(n, LOG, extra_local_load=extra)
    cap = n.generators_t.p_max_pu[["ls_a", "ls_b"]] * 100.0
    assert cap["ls_a"].tolist() == pytest.approx([15.0, 10.0, 10.0])
    assert cap["ls_b"].tolist() == pytest.approx([7.0, 7.0, 0.0])     # a flex-only bus can shed its draw
    apply_load_shedding_limits(n, LOG)
    assert (n.generators_t.p_max_pu["ls_b"] * 100.0).tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_held_flex_draw_by_bus_is_store_minus_dispatch():
    from scripts.market.solve_balancing import held_consumer_flex_draw_by_bus

    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    n.add("Bus", ["a", "b"])
    n.add("StorageUnit", ["ev", "dsr", "bat"], bus=["a", "a", "b"], carrier=[dc.EV_SMART_CHARGING, dc.IC_DSR, "Battery"],
          p_nom=10.0)
    ws = pd.DataFrame({"ev": [-4.0, 0.0], "dsr": [-1.0, 2.0], "bat": [-5.0, 5.0]}, index=n.snapshots)
    draw = held_consumer_flex_draw_by_bus(n, ws)
    assert list(draw.columns) == ["a"] and draw["a"].tolist() == pytest.approx([5.0, -2.0])


# ───────────── 5. BM rebuilt Stage 1 state of charge counts drains ─────────────

def test_stage1_state_of_charge_counts_the_drain():
    from scripts.market.solve_balancing import stage1_state_of_charge

    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=4, freq="h"))
    n.add("Bus", "a")
    n.add("StorageUnit", "ev", bus="a", carrier=dc.EV_SMART_CHARGING, p_nom=10.0, max_hours=4.0,
          efficiency_store=1.0, inflow=pd.Series([-2.0] * 4, index=n.snapshots))
    n.add("StorageUnit", "hydro", bus="a", carrier="hydro", p_nom=10.0, max_hours=4.0,
          inflow=pd.Series([3.0] * 4, index=n.snapshots))
    ws = pd.DataFrame({"ev": [-8.0, 0.0, 0.0, 0.0], "hydro": [0.0, 0.0, 0.0, 0.0]}, index=n.snapshots)
    soc = stage1_state_of_charge(n, ws)
    assert soc["ev"].tolist() == pytest.approx([6.0, 4.0, 2.0, 0.0])   # was 8, 8, 8, 8
    assert soc["hydro"].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])  # positive inflow left out, as before


# ───────────── 6. Validation demand line nets V2G and leaves flex out of storage ─────────────

def test_validation_demand_nets_v2g_and_storage_rows_exclude_flex():
    from scripts.CP30.energy_validation import network_energy_tables

    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    n.add("Bus", "a", carrier="AC")
    n.add("Load", "load_a", bus="a", p_set=100.0)
    n.add("Generator", "g", bus="a", carrier="CCGT", p_nom=500.0)
    n.add("StorageUnit", ["v2g", "dsr", "bat"], bus="a", carrier=[dc.V2G, dc.IC_DSR, "Battery"], p_nom=10.0)
    n.generators_t.p = pd.DataFrame({"g": [100.0, 100.0]}, index=n.snapshots)
    n.storage_units_t.p_store = pd.DataFrame({"v2g": [5.0, 0.0], "dsr": [2.0, 0.0], "bat": [3.0, 0.0]},
                                             index=n.snapshots)
    n.storage_units_t.p_dispatch = pd.DataFrame({"v2g": [0.0, 1.0], "dsr": [0.0, 2.0], "bat": [0.0, 2.0]},
                                                index=n.snapshots)
    t = network_energy_tables(n, include_dispatch=True)
    assert t["demand_mwh"] == pytest.approx(200.0 + (5.0 - 1.0) + (2.0 - 2.0))
    assert set(t["storage_dispatch_mwh"].index) == {"Battery"}


# ───────────── 7. Small consumer-flex units are not dropped ─────────────

def test_numerical_conditioning_keeps_small_flex_drains_as_loads():
    from scripts.solve.solve_network import improve_numerical_conditioning

    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    n.add("Bus", "a")
    n.add("StorageUnit", "tiny_sh", bus="a", carrier=dc.STORAGE_HEATER_FLEXIBILITY, p_nom=0.05,
          inflow=pd.Series([-0.01, -0.01], index=n.snapshots))
    n.add("StorageUnit", "tiny_bat", bus="a", carrier="Battery", p_nom=0.05)
    improve_numerical_conditioning(n, LOG)
    assert "tiny_sh" not in n.storage_units.index and "tiny_bat" not in n.storage_units.index
    # the drain's energy stays, as a fixed Load at the unit's bus
    assert n.loads.at["tiny_sh fixed drain", "bus"] == "a"
    assert n.loads_t.p_set["tiny_sh fixed drain"].tolist() == pytest.approx([0.01, 0.01])
