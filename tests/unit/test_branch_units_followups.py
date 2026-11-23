"""Follow-ups to the branch impedance-unit fix (27 Sep 2026).

Solve-time unit guard, impedance-preserving transformer relaxations, Appendix B
change rows with R = 0, unit-correct clustering, the HND register units
contract, the GB_network.xlsx repair override, the Stage 1 storage-churn
diagnostic, and a regression test on a built network's physical reactances.
"""

import logging

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.network_build.branch_units import (
    BRANCH_UNITS_META_KEY, UNITS_PU100, UNITS_PYPSA, BranchUnitsError,
    convert_network_pu100_to_pypsa, require_pypsa_branch_units, set_branch_units,
)
from scripts.network_build import ETYS_upgrades as U

LOG = logging.getLogger(__name__)


def _native():
    n = pypsa.Network()
    n.set_snapshots(range(2))
    for bus, v in [("A4", 400.0), ("B4", 400.0), ("D1", 132.0), ("E1", 132.0)]:
        n.add("Bus", bus, v_nom=v, x=0.0, y=0.0)
    n.add("Line", "L400", bus0="A4", bus1="B4", r=0.0008, x=0.01, b=0.28, s_nom=2000)
    n.add("Line", "L132", bus0="D1", bus1="E1", r=0.002, x=0.02, b=0.01, s_nom=150)
    n.add("Transformer", "T", bus0="B4", bus1="D1", r=0.0016, x=0.084, s_nom=240, model="pi")
    set_branch_units(n, UNITS_PU100)
    convert_network_pu100_to_pypsa(n, LOG)
    return n


# ── 1. guard ────────────────────────────────────────────────────────────────

def test_guard_refuses_network_without_units_flag():
    n = _native()
    n.meta = {"pypsa_version": "1.0.7"}  # a pre-27-Sep build
    with pytest.raises(BranchUnitsError, match="Rebuild the network"):
        require_pypsa_branch_units(n, "Stage 2 balancing solve", "ETYS", LOG)


def test_guard_accepts_native_and_branchless_networks():
    require_pypsa_branch_units(_native(), "solve", "ETYS", LOG)
    zonal = pypsa.Network()
    zonal.add("Bus", ["Z1", "Z2"])
    zonal.add("Link", "Z1-Z2", bus0="Z1", bus1="Z2", p_nom=100)
    require_pypsa_branch_units(zonal, "solve", "Zonal", LOG)


def test_reduced_build_converts_matpower_pu100(monkeypatch):
    from scripts.network_build.build_network import build_network_by_type
    n = build_network_by_type("Reduced", LOG)
    assert n.meta[BRANCH_UNITS_META_KEY] == UNITS_PYPSA
    raw = pd.read_csv("data/network/reduced_network/lines.csv", index_col=0)
    n.calculate_dependent_values()
    # PyPSA per unit on 1 MVA equals the MATPOWER pu (100 MVA base) / 100
    np.testing.assert_allclose(n.lines.x_pu.values, raw.x.values / 100.0, rtol=1e-12)
    require_pypsa_branch_units(n, "solve", "Reduced", LOG)


# ── 2. transformer relaxations preserve impedance ───────────────────────────

def test_copperplate_relaxation_preserves_transformer_impedance():
    from scripts.market.solve_wholesale import apply_copperplate_relaxation
    n = _native(); n.calculate_dependent_values(); x_pu = n.transformers.x_pu.copy()
    apply_copperplate_relaxation(n, {"transmission_relaxation": 1.0e6}, LOG)
    n.calculate_dependent_values()
    assert n.transformers.at["T", "s_nom"] == 1.0e6
    assert n.transformers.at["T", "s_nom_original"] == 240
    pd.testing.assert_series_equal(n.transformers.x_pu, x_pu)


def test_adequacy_relaxation_preserves_transformer_impedance():
    from scripts.market.adequacy_controls import apply_adequacy_relaxation
    n = _native(); n.calculate_dependent_values(); x_pu = n.transformers.x_pu.copy()
    audit = apply_adequacy_relaxation(n, {"relaxation": {"mode": "all_ac", "capacity_mw": 5000}}, LOG)
    n.calculate_dependent_values()
    assert n.transformers.at["T", "s_nom"] == 5000
    assert n.lines.at["L132", "s_nom"] == 5000
    pd.testing.assert_series_equal(n.transformers.x_pu, x_pu)
    assert int(audit["changed"].sum()) == 3


# ── 3. change rows apply a published R = 0 ──────────────────────────────────

def _change(node1, node2, r, x, ctype="OHL", ohl=10.0):
    return {"Node1": node1, "Node2": node2, "Year": 2029, "R (% on 100 MVA)": r,
            "X (% on 100 MVA)": x, "B (% on 100 MVA)": 0.0, "Winter Rating (MVA)": 2286.0,
            "Circuit Type": ctype, "OHL Length (km)": ohl, "Cable Length (km)": 0.0}


def test_circuit_change_applies_zero_r_and_keeps_placeholder_x():
    n = pypsa.Network()
    n.add("Bus", ["J1", "J2", "K1", "K2"], v_nom=132.0)
    n.add("Line", "J", bus0="J1", bus1="J2", r=0.002, x=0.02, s_nom=1000)
    n.add("Line", "K", bus0="K1", bus1="K2", r=0.002, x=0.02, s_nom=1000)
    ok, failed, _ = U.apply_circuit_changes(
        n, [_change("J1", "J2", 0.0, 0.0001, ohl=1.0),      # JUNB1--JUNV1- style: real R = 0
            _change("K1", "K2", 0.0, 0.0, ohl=12.0)], LOG)   # placeholder: keep R and X
    assert (ok, failed) == (2, 0)
    assert n.lines.at["J", "r"] == 0.0 and n.lines.at["J", "x"] == pytest.approx(1e-6)
    assert n.lines.at["K", "r"] == 0.002 and n.lines.at["K", "x"] == 0.02
    assert n.lines.at["K", "s_nom"] == 2286.0


def test_transformer_change_applies_zero_r():
    n = pypsa.Network()
    n.add("Bus", "T4", v_nom=400.0); n.add("Bus", "T1", v_nom=132.0)
    n.add("Transformer", "T", bus0="T4", bus1="T1", r=0.002, x=0.08, s_nom=240)
    rec = {"Node1": "T4", "Node2": "T1", "Year": 2029, "R (% on 100 MVA)": 0.0,
           "X (% on 100 MVA)": 7.5, "B (% on 100 MVA)": 0.0, "Winter Rating (MVA)": 240}
    U.apply_transformer_changes(n, [rec], LOG)
    assert n.transformers.at["T", "r"] == 0.0 and n.transformers.at["T", "x"] == pytest.approx(0.075)


# ── 4. clustering in per unit ───────────────────────────────────────────────

def _cluster_network():
    n = pypsa.Network()
    n.set_snapshots(range(1))
    for bus, v, x in [("A4", 400.0, 0.0), ("B4", 400.0, 1.0), ("A1", 132.0, 0.0), ("B1", 132.0, 1.0)]:
        n.add("Bus", bus, v_nom=v, x=x, y=0.0)
    # 400 kV 0.01 pu100 and a parallel 132 kV path 0.05 pu100 (both ohm)
    n.add("Line", "L4", bus0="A4", bus1="B4", x=0.01 * 400**2 / 100, r=0.0, s_nom=2000, length=50)
    n.add("Line", "L1", bus0="A1", bus1="B1", x=0.05 * 132**2 / 100, r=0.0, s_nom=200, length=50)
    # 400/132 kV transformers, 0.08 pu100 on 240 MVA -> x on own base 0.192
    n.add("Transformer", "TA", bus0="A4", bus1="A1", x=0.08 * 240 / 100, r=0.0, s_nom=240)
    n.add("Transformer", "TB", bus0="B4", bus1="B1", x=0.08 * 240 / 100, r=0.0, s_nom=240)
    n.meta = {BRANCH_UNITS_META_KEY: UNITS_PYPSA}
    return n


def test_transformer_to_line_conversion_keeps_per_unit_impedance():
    from scripts.network_clustering.cluster_network import NetworkClusterer
    n = NetworkClusterer()._convert_transformers_to_lines(_cluster_network())
    n.calculate_dependent_values()
    assert 100 * n.lines.at["tf_as_line_TA", "x_pu"] == pytest.approx(0.08)


def test_clustering_combines_parallel_paths_in_per_unit():
    from scripts.network_clustering.cluster_network import NetworkClusterer
    n = _cluster_network()
    busmap = {"A4": "A", "A1": "A", "B4": "B", "B1": "B"}
    c = NetworkClusterer().cluster_network(n, "busmap", {"busmap_source": busmap})
    c.calculate_dependent_values()
    assert len(c.lines) == 1
    # parallel 0.01 and 0.05 pu100 -> 0.008333 pu100, independent of voltage
    assert 100 * c.lines.x_pu.iloc[0] == pytest.approx(1.0 / (1 / 0.01 + 1 / 0.05))
    # input network left in ohm
    assert n.lines.at["L1", "x"] == pytest.approx(0.05 * 132**2 / 100)


def test_clustering_yaml_mean_strategy_is_overridden():
    from scripts.network_clustering.cluster_network import NetworkClusterer
    n = _cluster_network()
    busmap = {"A4": "A", "A1": "A", "B4": "B", "B1": "B"}
    c = NetworkClusterer().cluster_network(n, "busmap", {"busmap_source": busmap},
                                           aggregation_strategies={"line": {"x": "mean", "r": "mean"}})
    c.calculate_dependent_values()
    assert 100 * c.lines.x_pu.iloc[0] == pytest.approx(1.0 / (1 / 0.01 + 1 / 0.05))


# ── 5. HND register units ───────────────────────────────────────────────────

def test_hnd_line_reads_declared_reactance_not_coordinate(tmp_path):
    from scripts.CP30.hnd_hybrid import apply_hnd_hybrid_topology
    n = pypsa.Network()
    n.add("Bus", "ETYS_A", v_nom=400.0, x=0.0, y=0.0)
    n.add("Bus", "ETYS_B", v_nom=400.0, x=500000.0, y=0.0)
    set_branch_units(n, UNITS_PU100)
    reg = tmp_path / "assets.csv"
    pd.DataFrame([{
        "asset_id": "L", "component_type": "line", "name": "HND line", "bus0": "ETYS_A",
        "bus1": "ETYS_B", "x": 123456.0, "y": 0.0, "r_pu100": 0.0005, "x_pu100": 0.01,
        "b_pu100": 0.3, "s_nom": 2000.0, "in_service_year": 2030,
        "etys_overlap_status": "not_in_etys", "source_url": "https://example.test",
        "evidence_reference": "t",
    }]).to_csv(reg, index=False)
    apply_hnd_hybrid_topology(n, {"enabled": True, "asset_register": reg}, 2030,
                              tmp_path / "audit.csv", LOG)
    assert n.lines.at["HND line", "x"] == pytest.approx(0.01)


def test_hnd_register_with_duplicate_x_header_is_rejected(tmp_path):
    from scripts.CP30.hnd_hybrid import apply_hnd_hybrid_topology
    reg = tmp_path / "assets.csv"
    reg.write_text("asset_id,component_type,name,bus0,bus1,x,y,r,x,b,s_nom,in_service_year,"
                   "etys_overlap_status,source_url,evidence_reference\n"
                   "L,line,l,A,B,1,0,0.1,0.2,0,100,2030,not_in_etys,u,e\n")
    n = pypsa.Network(); n.add("Bus", ["A", "B"], v_nom=400.0)
    with pytest.raises(ValueError, match="duplicate"):
        apply_hnd_hybrid_topology(n, {"enabled": True, "asset_register": reg}, 2030,
                                  tmp_path / "audit.csv", LOG)


def test_repository_hnd_register_header_is_unambiguous():
    cols = pd.read_csv("data/network/HND/hnd_hybrid_asset_register.csv", nrows=0).columns
    assert {"r_pu100", "x_pu100", "b_pu100"} <= set(cols)
    assert not any("." in c for c in cols)


# ── 6. GB_network repair override ───────────────────────────────────────────

def test_beatrice_repair_transformer_takes_appendix_b_twin(tmp_path):
    from scripts.network_build.process_ETYS_data import repair_disconnected_components
    path = tmp_path / "gb.xlsx"
    pd.DataFrame([{
        "Node 1": "BEAT42", "Node 2": "BEAT22", "OHL Length (km)": 0.0, "Cable Length (km)": 0.001,
        "Circuit Type": "Transformer", "R (% on 100 MVA)": 0.173, "X (% on 100 MVA)": 16.03,
        "B (% on 100 MVA)": 0.0, "Winter Rating (MVA)": 320.0,
    }]).to_excel(path, sheet_name="AC", index=False)
    comps = pd.DataFrame(
        [["line", "AC", "MAIN41", "BEAT42", 0.0, 1e-4, 0.0, 500.0, 0.0],
         ["line", "AC", "BEAT22", "ISLE21", 0.0, 1e-4, 0.0, 500.0, 0.0],
         ["line", "AC", "MAIN41", "MAIN42", 0.0, 1e-4, 0.0, 500.0, 0.0]],
        columns=["component", "carrier", "bus0", "bus1", "r", "x", "b", "s_nom", "length_km"],
        index=["a", "b", "c"])
    out = repair_disconnected_components(comps, str(path), LOG)
    t = out[out.component == "transformer"].iloc[0]
    assert t.x == pytest.approx(0.05009) and t.r == pytest.approx(0.00054)


# ── 7. storage churn diagnostic ─────────────────────────────────────────────

def test_storage_churn_measures_simultaneous_charge_discharge(caplog):
    from scripts.market.solve_wholesale import measure_storage_churn, record_storage_churn
    n = pypsa.Network()
    n.set_snapshots(range(3))
    n.add("Bus", "A")
    n.add("StorageUnit", ["bat", "phs"], bus="A", p_nom=1000, efficiency_store=0.9,
          efficiency_dispatch=0.9)
    n.storage_units_t.p_store = pd.DataFrame({"bat": [500.0, 0.0, 800.0], "phs": [0.0, 10.0, 0.0]}, index=n.snapshots)
    n.storage_units_t.p_dispatch = pd.DataFrame({"bat": [400.0, 100.0, 900.0], "phs": [0.0, 0.0, 0.0]}, index=n.snapshots)
    with caplog.at_level(logging.WARNING):
        res = measure_storage_churn(n, LOG, label="w1", warn_mwh=100.0)
    # churned = 400 + 0 + 800 = 1200 MWh; loss = 1200 * (1/0.9 - 0.9)
    assert res["churned_mwh"] == pytest.approx(1200.0)
    assert res["dissipated_mwh"] == pytest.approx(1200.0 * (1 / 0.9 - 0.9))
    assert "storage churn" in caplog.text
    total = record_storage_churn(n, [res], LOG)
    assert n.meta["stage1_storage_churn"]["dissipated_mwh"] == pytest.approx(total, abs=1e-3)


# ── regression: a built network carries native units and physical x/km ─────

def test_built_network_has_native_units_and_physical_reactance_per_km(tmp_path):
    from scripts.network_build.ETYS_network import create_network
    # ETYS 2024 B-2-1 rows (% on 100 MVA): ABHA4A-EXET41 400 kV 48.785 km,
    # a 275 kV and a 132 kV OHL, plus a 400/132 kV SGT
    comps = pd.DataFrame(
        [["line", "AC", "ABHA4A", "EXET41", 0.000778, 0.009831, 0.279792, 2078.0, 48.785],
         ["line", "AC", "AAAA21", "BBBB21", 0.00076, 0.00846, 0.047, 1000.0, 20.0],
         ["line", "AC", "CCCC11", "DDDD11", 0.0057, 0.0446, 0.0016, 150.0, 20.0],
         ["line", "AC", "EXET41", "AAAA21", 0.0, 1e-4, 0.0, 9999.0, 0.0],
         ["transformer", "AC", "EXET41", "CCCC11", 0.001658, 0.084025, -0.015604, 275.0, 0.0]],
        columns=["component", "carrier", "bus0", "bus1", "r", "x", "b", "s_nom", "length_km"],
        index=["L400", "L275", "L132", "C", "T"])
    buses = pd.DataFrame({"v_nom": [400.0, 400.0, 275.0, 275.0, 132.0, 132.0], "carrier": "AC",
                          "lon": [-3.5, -3.4, -2.0, -2.1, -1.0, -1.1],
                          "lat": [50.7, 50.8, 53.0, 53.1, 52.0, 52.1]},
                         index=["ABHA4A", "EXET41", "AAAA21", "BBBB21", "CCCC11", "DDDD11"])
    path = tmp_path / "net.nc"
    create_network(comps, buses, LOG, export_path=str(path), etys_upgrades_enabled=False)
    n = pypsa.Network(str(path))
    assert n.meta[BRANCH_UNITS_META_KEY] == UNITS_PYPSA
    require_pypsa_branch_units(n, "regression", "ETYS", LOG)
    ohm_per_km = n.lines.x / n.lines.length_km.replace(0, np.nan)
    assert 0.25 < ohm_per_km["L400"] < 0.40      # 400 kV OHL ~0.32 ohm/km
    assert 0.25 < ohm_per_km["L275"] < 0.40
    assert 0.25 < ohm_per_km["L132"] < 0.45
    n.calculate_dependent_values()
    assert 1.5e-4 < 100 * n.lines.x_pu["L400"] / 48.785 < 2.5e-4   # pu100 per km
    assert 100 * n.transformers.x_pu["T"] == pytest.approx(0.084025)
    assert 100 * n.lines.x_pu["C"] == pytest.approx(1e-4)
    # every branch carries seasonal ratings and a licensee after the build
    for col in ("s_nom_winter", "s_nom_spring", "s_nom_summer", "s_nom_autumn",
                "seasonal_rating_source", "etys_licensee"):
        assert col in n.lines.columns and n.lines[col].notna().all()
    assert n.lines.at["C", "seasonal_rating_source"] == "relaxed"
    assert (n.transformers.seasonal_rating_source == "winter_only").all()
