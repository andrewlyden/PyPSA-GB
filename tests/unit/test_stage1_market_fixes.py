"""Stage 1 market fixes of 27 Sep 2026 (market fix list items D5, C1, C2, C3, C5, C6, C12).

* D5: the hydrogen store's level is carried across rolling windows (it was
  ``e_cyclic`` per window, so hydrogen could never move between days), with a
  month-end condition like the storage units' one.
* C1: a central dispatch Stage 1 enforces the secured boundary limits.
* C2: the existing pumped-hydro stations, which CP30 books as LAES, get the
  pumped-hydro state-of-charge floors.
* C3: the copperplate lifts GB-internal island HVDC limits (Shetland).
* C5, C6, C12: config and workflow declarations.
"""

import copy
import logging
from pathlib import Path

import pandas as pd
import pypsa
import pytest
import yaml

from scripts.market.market_utils import (
    apply_carried_store_levels,
    build_month_end_store_callback,
    carried_store_names,
    final_store_levels,
    prepare_carried_stores,
    resolve_store_level_config,
)
from scripts.solve.hydro_constraints import build_hydro_constraints_callback, get_pumped_hydro_names
from config.config_loader import load_defaults

ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger("test_stage1_market_fixes")


def _defaults():
    return load_defaults()


# ── D5: hydrogen store level carried across windows ─────────────────────────

def _hydrogen_network(hours=48, windy_day=1):
    """Cheap wind on one day (day 1 by default), only dear gas on the other; an H2 turbine and a store."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=hours, freq="h"))
    n.add("Bus", "el", carrier="AC")
    n.add("Bus", "GB_H2", carrier="H2")
    wind = [200.0 if (i < 24) == (windy_day == 1) else 0.0 for i in range(hours)]
    n.add("Generator", "wind", bus="el", p_nom=1.0, p_max_pu=pd.Series(wind, index=n.snapshots),
          marginal_cost=0.0, carrier="wind_onshore")
    n.add("Generator", "gas", bus="el", p_nom=500.0, marginal_cost=150.0, carrier="CCGT")
    n.add("Load", "load", bus="el", p_set=100.0)
    n.add("Link", "electrolysis", bus0="el", bus1="GB_H2", p_nom=100.0, efficiency=0.7, carrier="electrolysis")
    n.add("Link", "h2_turbine", bus0="GB_H2", bus1="el", p_nom=200.0, efficiency=0.5,
          marginal_cost=5.0, carrier="H2_turbine")
    n.add("Store", "GB_H2_storage", bus="GB_H2", carrier="H2_gas", e_nom=4000.0, e_cyclic=True,
          standing_loss=0.0)
    return n


def _rolling(n, market_config, window=24):
    """The Stage 1 loop's store handling, on a toy network (HiGHS)."""
    start = prepare_carried_stores(n, market_config, LOG)
    names = carried_store_names(n, market_config)
    callback = build_month_end_store_callback(n, market_config, start, n.snapshots[-1], LOG)
    levels, trajectory, turbine = None, [], []
    for i in range(0, len(n.snapshots), window):
        day = n.copy()
        day.set_snapshots(n.snapshots[i:i + window])
        apply_carried_store_levels(day, levels)
        status, condition = day.optimize(solver_name="highs", extra_functionality=callback)
        assert status == "ok", condition
        levels = final_store_levels(day, names) if len(names) else None
        trajectory.append(day.stores_t.e["GB_H2_storage"].copy())
        turbine.append(day.links_t.p0["h2_turbine"].copy())
    _rolling.turbine = pd.concat(turbine)
    return start, pd.concat(trajectory)


def test_store_config_defaults_match_defaults_yaml():
    block = _defaults()["market"]["hydrogen_store"]
    resolved = resolve_store_level_config({})
    assert resolved == {
        "carry_level": block["carry_level"],
        "initial_fraction": block["initial_fraction"],
        "month_end": block["month_end"],
        "carriers": tuple(block["carriers"]),
        "end_of_window_value_gbp_per_mwh_h2": block["end_of_window_value_gbp_per_mwh_h2"],
    }
    with pytest.raises(ValueError):
        resolve_store_level_config({"hydrogen_store": {"initial_fraction": 1.5}})
    with pytest.raises(ValueError):
        resolve_store_level_config({"hydrogen_store": {"month_end": "cyclic"}})


def test_prepare_switches_the_store_from_cyclic_to_a_carried_start():
    n = _hydrogen_network()
    start = prepare_carried_stores(n, {}, LOG)
    assert start.to_dict() == {"GB_H2_storage": pytest.approx(2000.0)}
    assert not bool(n.stores.at["GB_H2_storage", "e_cyclic"])
    off = _hydrogen_network()
    assert prepare_carried_stores(off, {"hydrogen_store": {"carry_level": False}}, LOG).empty
    assert bool(off.stores.at["GB_H2_storage", "e_cyclic"])


def test_the_store_level_carries_into_the_next_window_and_is_burnt_there():
    n = _hydrogen_network()
    # Month end free so day 2 may use what the run started with.
    start, e = _rolling(n, {"hydrogen_store": {"month_end": "free"}})
    end_day1 = float(e.iloc[23])
    assert end_day1 == pytest.approx(float(start.iloc[0]), abs=1e-3)   # windy day 1 leaves it alone
    assert float(e.iloc[-1]) < end_day1 - 100.0                        # still day 2 burns it against gas
    burnt = float(_rolling.turbine.iloc[24:].sum())
    assert burnt == pytest.approx(end_day1 - float(e.iloc[-1]), rel=1e-4)


def test_per_window_cyclic_store_could_not_move_hydrogen_between_days():
    # The defect: with the built e_cyclic applied per window, day 2 (no wind)
    # burns only what it makes itself, which against 150 GBP gas is nothing.
    _rolling(_hydrogen_network(), {"hydrogen_store": {"carry_level": False}})
    cyclic_day2 = float(_rolling.turbine.iloc[24:].sum())
    _rolling(_hydrogen_network(), {"hydrogen_store": {"month_end": "free"}})
    carried_day2 = float(_rolling.turbine.iloc[24:].sum())
    assert cyclic_day2 == pytest.approx(0.0, abs=1e-3)
    assert carried_day2 > 100.0


def test_month_end_holds_the_store_at_or_above_its_start():
    n = _hydrogen_network()
    n.links.loc["electrolysis", "p_nom"] = 400.0   # 280 MW of hydrogen: the last day can refill the start
    start, e = _rolling(n, {})
    assert float(e.iloc[-1]) >= float(start.iloc[0]) - 1e-3


def test_month_end_target_is_capped_at_what_the_electrolysers_can_make():
    # Still day 1 burns hydrogen (no terminal value in a myopic window); the
    # windy last day can add only 0.95 x 0.7 x 24 MWh with a 1 MW electrolyser.
    n = _hydrogen_network(windy_day=2)
    n.links.loc["electrolysis", "p_nom"] = 1.0
    start, e = _rolling(n, {})
    target = min(float(start.iloc[0]), float(e.iloc[23]) + 0.95 * 0.7 * 24)
    assert target < float(start.iloc[0])                     # capped, and the window still solved
    assert float(e.iloc[-1]) >= target - 1e-3


# ── C2: existing pumped hydro booked as LAES gets the pumped-hydro floors ───

def _pumped_network():
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=4, freq="h"))
    n.add("Bus", "b", carrier="AC")
    n.add("Generator", "gas", bus="b", p_nom=1000.0, marginal_cost=100.0, carrier="CCGT")
    n.add("Load", "load", bus="b", p_set=100.0)
    for name in ("CP30_LAES_existing_Dinorwig_569", "CP30_LAES_4_573"):
        n.add("StorageUnit", name, bus="b", p_nom=100.0, max_hours=4.0, carrier="LAES",
              state_of_charge_initial=200.0, efficiency_store=1.0, efficiency_dispatch=1.0,
              marginal_cost=1.0, cyclic_state_of_charge=False)
    return n


HYDRO_CONFIG = {
    "renewables": {
        "hydro": {
            "enabled": True,
            "pumped_hydro": {
                "min_soc_fraction": 0.15,
                "wholesale_min_soc_fraction": 0.40,
                "existing_sites": [{"name": "Dinorwig", "site_code": "DINO"}, {"name": "Cruachan"}],
            },
        }
    }
}


def test_existing_pumped_hydro_is_found_by_name_and_by_attribute():
    n = _pumped_network()
    assert get_pumped_hydro_names(n, HYDRO_CONFIG["renewables"]["hydro"]) == ["CP30_LAES_existing_Dinorwig_569"]
    assert get_pumped_hydro_names(n) == []                  # carrier LAES alone is not pumped hydro
    n.add("StorageUnit", "somewhere", bus="b", p_nom=10.0, max_hours=2.0, carrier="LAES")
    n.storage_units["existing_site"] = ""
    n.storage_units.loc["somewhere", "existing_site"] = "Cruachan"
    assert set(get_pumped_hydro_names(n, HYDRO_CONFIG["renewables"]["hydro"])) == {"CP30_LAES_existing_Dinorwig_569", "somewhere"}


@pytest.mark.parametrize("override, floor", [(0.40, 160.0), (None, 60.0)])
def test_the_floor_applies_to_the_existing_site_and_not_to_new_build_ldes(override, floor):
    n = _pumped_network()
    callback = build_hydro_constraints_callback(n, HYDRO_CONFIG, pumped_min_soc_override=override)
    status, condition = n.optimize(solver_name="highs", extra_functionality=callback)
    assert status == "ok", condition
    soc = n.storage_units_t.state_of_charge
    assert soc["CP30_LAES_existing_Dinorwig_569"].min() >= floor - 1e-4
    assert soc["CP30_LAES_4_573"].min() < 1.0             # new-build LDES drains against gas


# ── C3: island HVDC relaxed in the copperplate ──────────────────────────────

def _island_network():
    from scripts.market.solve_wholesale import relax_island_hvdc_links  # noqa: F401  (import check)

    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    for bus in ("M1", "M2", "KERG1J", "KERG2", "HVDC_External_Norway"):
        n.add("Bus", bus, carrier="AC", v_nom=400.0)
    n.add("Bus", "Central DC Substation", carrier="DC", v_nom=400.0)
    n.add("Line", "m", bus0="M1", bus1="M2", x=0.1, s_nom=1000.0)
    n.add("Line", "shetland", bus0="KERG1J", bus1="KERG2", x=0.1, s_nom=1000.0)
    n.add("Line", "m2", bus0="M1", bus1="M2", x=0.1, s_nom=1000.0)
    n.add("Load", "mainland", bus="M1", p_set=500.0)
    n.add("Load", "shetland", bus="KERG2", p_set=20.0)
    n.add("Generator", "viking", bus="KERG2", p_nom=850.0, carrier="wind_onshore")
    n.add("Generator", "ls", bus="KERG2", p_nom=400.0, carrier="load_shedding", marginal_cost=6000.0)
    n.add("Link", "KERG1J_Central DC Substation_0", bus0="KERG1J", bus1="Central DC Substation",
          p_nom=600.0, p_min_pu=-1.0, carrier="DC")
    n.add("Link", "SPIT_Central", bus0="M2", bus1="Central DC Substation", p_nom=800.0, p_min_pu=-1.0, carrier="DC")
    n.add("Link", "M1_M2_HVDC", bus0="M1", bus1="M2", p_nom=2000.0, p_min_pu=-1.0, carrier="DC")
    n.add("Link", "IC_NSL", bus0="HVDC_External_Norway", bus1="M1", p_nom=1400.0, p_min_pu=-1.0, carrier="DC")
    return n


def test_island_hvdc_is_lifted_for_the_market_and_restored_after():
    from scripts.market.solve_wholesale import relax_island_hvdc_links, restore_island_hvdc_links

    n = _island_network()
    relaxed = relax_island_hvdc_links(n, {}, LOG)
    assert set(relaxed) == {"KERG1J_Central DC Substation_0", "SPIT_Central"}
    headroom = 1.05 * (850.0 + 20.0)
    assert n.links.at["KERG1J_Central DC Substation_0", "p_nom"] == pytest.approx(headroom)
    assert n.links.at["SPIT_Central", "p_nom"] == pytest.approx(headroom)
    assert n.links.at["M1_M2_HVDC", "p_nom"] == 2000.0      # mainland-internal: the collapse drops it anyway
    assert n.links.at["IC_NSL", "p_nom"] == 1400.0          # interconnectors keep their ratings
    restore_island_hvdc_links(n, relaxed, LOG)
    assert n.links.at["KERG1J_Central DC Substation_0", "p_nom"] == 600.0
    assert n.links.at["SPIT_Central", "p_nom"] == 800.0


def test_island_hvdc_relaxation_can_be_switched_off():
    from scripts.market.solve_wholesale import relax_island_hvdc_links

    n = _island_network()
    assert relax_island_hvdc_links(n, {"relax_island_hvdc": False}, LOG) == []
    assert n.links.at["KERG1J_Central DC Substation_0", "p_nom"] == 600.0


def test_island_hvdc_defaults_in_yaml():
    wholesale = _defaults()["market"]["wholesale"]
    assert wholesale["relax_island_hvdc"] is True and wholesale["island_hvdc_carriers"] == ["DC"]


# ── C1: central dispatch Stage 1 enforces the secured boundaries ────────────

def test_central_stage1_builds_the_boundary_callback_and_copperplate_does_not(monkeypatch):
    import scripts.market.solve_wholesale as sw

    calls = []
    monkeypatch.setattr(sw, "_build_neso_boundary_constraints_callback",
                        lambda network, scenario_config, logger: calls.append(1) or (lambda n, s: None))
    assert sw.build_central_boundary_callback(None, {}, {"copperplate_mode": "collapse"}, LOG) is None
    assert calls == []
    assert sw.build_central_boundary_callback(None, {}, {"copperplate_mode": "none"}, LOG) is not None
    assert calls == [1]


# ── C5, C6, C12: config and workflow declarations ───────────────────────────

SENSITIVITIES = ["markup40", "markup20", "market_uc", "windbid_low", "windbid_high"]


def test_the_ten_sensitivities_share_their_month_stage1_and_pass_the_snakefile_guard():
    import sys

    sys.path.insert(0, str(ROOT / "config"))
    from config_loader import load_config

    scenarios = load_config(include_archive=True)["scenarios"]   # the sensitivities are archived (D9)
    src = (ROOT / "Snakefile").read_text(encoding="utf-8")
    namespace = {"scenarios": scenarios}
    exec(src[src.index("WHOLESALE_FROM_ALLOWED_DIFFERENCES ="):src.index("_network_from_parents_list = ")], namespace)
    ids = []
    for kind in SENSITIVITIES:
        for month in ("Jan", "Jul"):
            sid = f"CP30_FFR_2030_ETYS_market_{month}2030_{kind}"
            parent = f"CP30_FFR_2030_ETYS_market_{month}2030_neso_es5_landing_reassignment_reinforced"
            assert scenarios[sid]["wholesale_from"] == parent
            ids.append(sid)
    namespace["_wholesale_from_parents"](ids)   # raises if anything outside market.balancing differs


def test_stage1_rule_declares_scarcity_curve_and_central_boundary_inputs_and_priority():
    text = (ROOT / "rules" / "market.smk").read_text(encoding="utf-8")
    assert "unpack(_wholesale_extra_inputs)" in text
    assert 'inputs["scarcity_curve"]' in text
    assert "use rule solve_wholesale_market as solve_wholesale_market_central with:" in text
    assert "priority: CENTRAL_WHOLESALE_PRIORITY" in text
