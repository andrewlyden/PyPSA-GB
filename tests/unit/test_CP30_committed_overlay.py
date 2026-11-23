"""Net Zero Teesside Power as a declared overlay on CP30 (user decision, 23 Sep 2026)."""

from pathlib import Path

import pandas as pd
import pypsa
import pytest

from scripts.CP30.committed_overlay import (
    DATA_SOURCE, apply_committed_overlay, load_committed_overlay, overlay_capacity_by_carrier,
)

OVERLAY = Path("data/market/CP30/committed_overlay_2030.csv")


def _scaled():
    return pd.DataFrame({"fuel_type": ["CCGT"], "technology": ["CCGT"], "capacity_mw": [31253.35],
                         "bus": ["X"], "data_source": ["CP30_gas_station_schedule"]})


def test_the_overlay_is_net_zero_teesside_at_tod_point_on_its_dpa():
    overlay = load_committed_overlay(OVERLAY)
    row = overlay.set_index("station_name").loc["Net Zero Teesside Power"]
    assert row["carrier"] == "CCS Gas" and row["capacity_mw"] == pytest.approx(742.0)
    assert row["bus"] == "TODP21" and row["support_type"] == "DPA"
    assert overlay_capacity_by_carrier(OVERLAY) == {"CCS Gas": pytest.approx(742.0)}


def test_overlay_appends_plant_cp30_omits_and_refuses_double_counting():
    targets = pd.DataFrame({"carrier": ["CCGT", "CCS Gas"], "capacity_mw": [31253.35, 0.0]})
    out = apply_committed_overlay(_scaled(), OVERLAY, targets)
    nzt = out[out["data_source"].eq(DATA_SOURCE)]
    assert len(nzt) == 1 and nzt["fuel_type"].iloc[0] == "CCS Gas" and nzt["bus"].iloc[0] == "TODP21"
    assert out.loc[out["fuel_type"].eq("CCGT"), "capacity_mw"].sum() == pytest.approx(31253.35)
    with pytest.raises(ValueError, match="already gives capacity"):
        apply_committed_overlay(_scaled(), OVERLAY, pd.DataFrame({"carrier": ["CCS Gas"], "capacity_mw": [3310.0]}))


def test_dpa_plant_is_dispatched_ahead_of_unabated_gas():
    from scripts.generators.apply_marginal_costs import _apply_dpa_dispatch

    n = pypsa.Network()
    n.add("Bus", "b")
    n.add("Generator", "ccgt_a", bus="b", carrier="CCGT", p_nom=500, marginal_cost=70.0)
    n.add("Generator", "ccgt_b", bus="b", carrier="CCGT", p_nom=500, marginal_cost=85.0)
    n.add("Generator", "nzt", bus="b", carrier="CCS Gas", p_nom=742, marginal_cost=90.0)
    n.generators["support_type"] = ["merchant", "merchant", "DPA"]
    mc = pd.DataFrame({"generator": ["ccgt_a", "ccgt_b", "nzt"], "carrier": ["CCGT", "CCGT", "CCS Gas"],
                       "marginal_cost": [70.0, 85.0, 90.0], "source": ["formula"] * 3})
    out = _apply_dpa_dispatch(n, mc, {"dpa_margin_gbp_per_mwh": 1.0}).set_index("generator")
    assert out.loc["nzt", "marginal_cost"] == pytest.approx(69.0)
    assert n.generators.at["nzt", "marginal_cost"] == pytest.approx(69.0)
    # A DPA plant already cheaper than unabated gas keeps its own cost.
    mc.loc[2, "marginal_cost"] = 60.0
    n.generators.at["nzt", "marginal_cost"] = 60.0
    out = _apply_dpa_dispatch(n, mc, {"dpa_margin_gbp_per_mwh": 1.0}).set_index("generator")
    assert out.loc["nzt", "marginal_cost"] == pytest.approx(60.0)


def test_validation_checks_the_overlay_on_its_own_row():
    from scripts.CP30.validation import validate_cp30_component_totals

    es1 = Path("resources/CP30/CP30_ES1.csv")
    ed1 = Path("resources/CP30/CP30_ED1.csv")
    if not (es1.exists() and ed1.exists()):
        pytest.skip("CP30 resources not extracted")
    generators = pd.DataFrame({"carrier": ["CCS Gas"], "p_nom": [742.0]})
    table = validate_cp30_component_totals(es1, ed1, "Further Flex and Renewables", 2030,
                                           generators=generators, overlay_capacity={"CCS Gas": 742.0})
    row = table[table["source"].eq("committed_overlay")]
    assert len(row) == 1 and row["status"].iloc[0] == "PASS"


def test_dpa_plant_holds_no_capacity_market_agreement():
    from scripts.market.capacity_market import attach_capacity_market_revenue

    rates = pd.DataFrame({"carrier": ["CCGT"], "gbp_per_kw_connection_year": [40.0], "connection_mw_2029_30": [30000.0]})
    assets = pd.DataFrame({"carrier": ["CCGT", "CCS Gas"], "support_type": ["DPA", "DPA"], "capacity_mw": [742.0, 742.0]})
    assert attach_capacity_market_revenue(assets, rates)["capacity_market_revenue_gbp"].sum() == 0.0


# ── Carve-out (user decision, 14 Oct 2026: capacities equal CP30's) ──────────

CARVE = {"CCS Gas": {"cp30_line": "gas", "from_carrier": "CCGT", "reason": "test"}}
GAS_TARGETS = pd.DataFrame({"carrier": ["gas", "CHP"], "capacity_mw": [31253.35, 3746.65],
                            "split_carriers": [("CCGT", "OCGT", "gas_engine"), ()]})


def _gas_table():
    return pd.DataFrame({"fuel_type": ["CCGT", "CCGT", "OCGT", "gas_engine", "CHP"],
                         "technology": ["CCGT", "CCGT", "OCGT", "gas_engine", "CHP"],
                         "capacity_mw": [20000.0, 3377.95, 4143.11, 3732.29, 3746.65],
                         "bus": list("ABCDE"), "gsp": list("vwxyz"), "data_source": ["CP30"] * 5})


def test_carve_out_config_is_validated():
    from scripts.CP30.committed_overlay import carved_line_by_carrier, resolve_overlay_carve_out

    assert resolve_overlay_carve_out(None) == {}
    assert carved_line_by_carrier(resolve_overlay_carve_out(CARVE)) == {"CCS Gas": "gas"}
    with pytest.raises(ValueError, match="needs exactly"):
        resolve_overlay_carve_out({"CCS Gas": {"cp30_line": "gas", "from_carrier": "CCGT"}})
    with pytest.raises(ValueError, match="every field"):
        resolve_overlay_carve_out({"CCS Gas": {"cp30_line": "gas", "from_carrier": "CCGT", "reason": " "}})


def test_carve_out_takes_the_overlay_out_of_ccgt_so_total_gas_is_cp30s():
    from scripts.CP30.committed_overlay import apply_overlay_carve_out, resolve_overlay_carve_out

    carve = resolve_overlay_carve_out(CARVE)
    out = apply_overlay_carve_out(_gas_table(), OVERLAY, carve, GAS_TARGETS)
    out = apply_committed_overlay(out, OVERLAY, GAS_TARGETS, carve_out=carve)
    by = out.groupby("fuel_type")["capacity_mw"].sum()
    assert by["CCGT"] == pytest.approx(23377.95 - 742.0)
    assert by[["OCGT", "gas_engine", "CHP"]].tolist() == pytest.approx([4143.11, 3732.29, 3746.65])
    assert by[["CCGT", "OCGT", "gas_engine", "CCS Gas"]].sum() == pytest.approx(31253.35)
    assert by.sum() == pytest.approx(35000.0)
    nzt = out[out["data_source"].eq(DATA_SOURCE)].iloc[0]
    assert nzt["capacity_mw"] == pytest.approx(742.0) and nzt["support_type"] == "DPA"
    assert nzt["overlay_capacity_basis"] == "declared_overlay_carved_from_cp30_CCGT"


def test_carve_out_refuses_a_carrier_outside_the_line_or_too_much():
    from scripts.CP30.committed_overlay import apply_overlay_carve_out

    bad = {"CCS Gas": {"cp30_line": "gas", "from_carrier": "CHP", "reason": "x"}}
    with pytest.raises(ValueError, match="not a member"):
        apply_overlay_carve_out(_gas_table(), OVERLAY, bad, GAS_TARGETS)
    small = _gas_table()
    small.loc[small["fuel_type"].eq("CCGT"), "capacity_mw"] = [300.0, 100.0]
    with pytest.raises(ValueError, match="exceeds"):
        apply_overlay_carve_out(small, OVERLAY, CARVE, GAS_TARGETS)


def test_carve_out_with_the_station_schedule_retires_the_least_evidenced_ccgt():
    from scripts.CP30.committed_overlay import apply_overlay_carve_out
    from scripts.CP30.gas_schedule import apply_gas_station_schedule

    schedule = Path("data/market/CP30/gas_station_schedule_2030.csv")
    efficiency = {"CCGT": 0.52, "OCGT": 0.38, "gas_engine": 0.35}

    def fleet(table):
        targets = table[table["fuel_type"].isin(["CCGT", "OCGT", "gas_engine"])].groupby("fuel_type")["capacity_mw"].sum()
        out = apply_gas_station_schedule(table, schedule, targets.to_dict(), efficiency)
        return out[out["fuel_type"].eq("CCGT")].set_index("station_name")["capacity_mw"]

    before = fleet(_gas_table())
    after = fleet(apply_overlay_carve_out(_gas_table(), OVERLAY, CARVE, GAS_TARGETS))
    assert before.sum() - after.sum() == pytest.approx(742.0)
    change = after.reindex(before.index).fillna(0.0) - before
    moved = change[change.abs() > 1e-6].round(2).to_dict()
    # Retention order (15 Oct 2026): the 742 MW comes off the oldest stations holding
    # only a one-year 2029/30 Capacity Market agreement (both 1998; ties go by name).
    assert moved == {"Rocksavage Power Company Ltd": -190.45, "Didcot B": -551.55}


def test_validation_counts_a_carved_overlay_inside_cp30_gas():
    from scripts.CP30.validation import validate_cp30_component_totals

    es1 = Path("resources/CP30/CP30_ES1.csv")
    ed1 = Path("resources/CP30/CP30_ED1.csv")
    if not (es1.exists() and ed1.exists()):
        pytest.skip("CP30 resources not extracted")
    generators = pd.DataFrame({"carrier": ["CCGT", "OCGT", "gas_engine", "CCS Gas"],
                               "p_nom": [22635.95, 4143.11, 3732.29, 742.0]})
    kwargs = dict(generators=generators, overlay_capacity={"CCS Gas": 742.0})
    carved = validate_cp30_component_totals(es1, ed1, "Further Flex and Renewables", 2030,
                                            overlay_carved_into={"CCS Gas": "gas"}, **kwargs).set_index("item")
    assert carved.loc["gas", "status"] == "PASS"
    assert carved.loc["CCS Gas (committed overlay, carved out of CP30 gas)", "status"] == "PASS"
    # Without the declaration the same fleet is 742 MW short of CP30's gas line.
    plain = validate_cp30_component_totals(es1, ed1, "Further Flex and Renewables", 2030, **kwargs).set_index("item")
    assert plain.loc["gas", "status"] != "PASS"
    with pytest.raises(ValueError, match="no overlay capacity"):
        validate_cp30_component_totals(es1, ed1, "Further Flex and Renewables", 2030, generators=generators,
                                       overlay_carved_into={"CCS Gas": "gas"})


def test_ffr_2030_carves_nzt_out_of_ccgt_and_new_dispatch_does_not():
    import sys

    sys.path.insert(0, str(Path("config").resolve()))
    from config_loader import load_config
    from scripts.CP30.committed_overlay import resolve_overlay_carve_out

    scenarios = load_config(include_archive=True)["scenarios"]   # CP30 New Dispatch is archived
    ffr = scenarios["CP30_FFR_2030"]["fleet"]["spatial_proxy"]
    carve = resolve_overlay_carve_out(ffr.get("committed_overlay_carve_out"))
    assert carve["CCS Gas"]["cp30_line"] == "gas" and carve["CCS Gas"]["from_carrier"] == "CCGT"
    nd = scenarios["CP30_ND_2030_ETYS_market_neso_es5_landing_reassignment_reinforced"]["fleet"]["spatial_proxy"]
    assert not nd.get("committed_overlay") and not nd.get("committed_overlay_carve_out")
