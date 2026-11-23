"""Branch impedance units: ETYS pu on 100 MVA -> PyPSA native units, exactly once.

Covers the unit conversion (lines ohm/siemens at bus0 v_nom, transformers pu
on own s_nom), the zero/missing impedance policy shared by the base sheets and
the Appendix B upgrades (item N1), the CP30 accelerated components (item N2),
and the BM voltage floor that relaxes transformers into sub-220 kV networks
with their lines while preserving transformer impedance.
"""

import logging

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.network_build.branch_units import (
    BRANCH_UNITS_META_KEY,
    UNITS_PU100,
    UNITS_PYPSA,
    branch_values_for_network,
    branch_x_pu100,
    convert_network_pu100_to_pypsa,
    convert_network_pypsa_to_pu100,
    get_branch_units,
    resolve_source_impedance_pu100,
    set_branch_units,
    set_transformer_s_nom_preserving_impedance,
)
from scripts.network_build.etys_file_registry import ELECTRICAL_DEFAULTS
from scripts.network_build import ETYS_upgrades as U
from scripts.CP30.reinforcement_layer import apply_cp30_reinforcement_layer
from scripts.solve.solve_network import (
    apply_line_rating_overrides,
    bm_voltage_floor_branches,
    improve_numerical_conditioning,
)

LOG = logging.getLogger(__name__)


def _pu100_network():
    """400/275/132/33 kV toy network with ETYS-style pu100 branch values."""
    n = pypsa.Network()
    n.set_snapshots(range(1))
    for bus, v in [("A4", 400), ("B4", 400), ("C2", 275), ("D1", 132), ("E1", 132), ("F3", 33)]:
        n.add("Bus", bus, v_nom=v)
    # 400 kV OHL, 48.8 km: ETYS ABHA4A-EXET41 row (R 0.0778 %, X 0.9831 %, B 27.9792 %)
    n.add("Line", "L400", bus0="A4", bus1="B4", r=0.000778, x=0.009831, b=0.279792, s_nom=2078)
    n.add("Line", "L132", bus0="D1", bus1="E1", r=0.002, x=0.02, b=0.01, s_nom=150)
    n.add("Line", "L400_132", bus0="B4", bus1="E1", r=0.002, x=0.02, b=0.5, s_nom=500)  # BMU-edge style
    n.add("Transformer", "T400_275", bus0="A4", bus1="C2", r=0.0002, x=0.016, b=0.0, s_nom=1000, model="pi")
    n.add("Transformer", "T400_132", bus0="B4", bus1="D1", r=0.001658, x=0.084025, b=-0.015604,
          s_nom=275, model="pi")
    n.add("Transformer", "T132_33", bus0="D1", bus1="F3", r=0.01, x=0.25, b=0.0, s_nom=60, model="pi")
    set_branch_units(n, UNITS_PU100)
    return n


def test_line_conversion_matches_pypsa_convention():
    n = _pu100_network()
    convert_network_pu100_to_pypsa(n, LOG)
    # x_ohm = x_pu100 * V^2 / 100 ; b_S = b_pu100 * 100 / V^2
    assert n.lines.at["L400", "x"] == pytest.approx(0.009831 * 400**2 / 100)   # 15.73 ohm
    assert n.lines.at["L400", "b"] == pytest.approx(0.279792 * 100 / 400**2)   # 1.75e-4 S
    assert n.lines.at["L132", "x"] == pytest.approx(0.02 * 132**2 / 100)
    n.calculate_dependent_values()
    # PyPSA per unit on 1 MVA = pu100 / 100, for every line (bus0 v_nom, incl. mixed-voltage)
    assert n.lines.at["L400", "x_pu"] == pytest.approx(0.009831 / 100)
    assert n.lines.at["L400", "b_pu"] == pytest.approx(0.279792 * 100)
    assert n.lines.at["L132", "x_pu"] == pytest.approx(0.02 / 100)
    assert n.lines.at["L400_132", "x_pu"] == pytest.approx(0.02 / 100)
    # 400 kV OHL is ~0.32 ohm/km and 2e-4 pu100/km
    assert n.lines.at["L400", "x"] / 48.785 == pytest.approx(0.322, rel=0.01)


def test_transformer_conversion_matches_pypsa_convention():
    n = _pu100_network()
    convert_network_pu100_to_pypsa(n, LOG)
    # x (pu on own s_nom) = x_pu100 * s_nom / 100
    assert n.transformers.at["T400_132", "x"] == pytest.approx(0.084025 * 275 / 100)
    assert n.transformers.at["T400_132", "b"] == pytest.approx(-0.015604 * 100 / 275)
    n.calculate_dependent_values()
    for name, x100 in [("T400_275", 0.016), ("T400_132", 0.084025), ("T132_33", 0.25)]:
        assert n.transformers.at[name, "x_pu"] == pytest.approx(x100 / 100)
    assert n.transformers.at["T400_132", "b_pu"] == pytest.approx(-0.015604 * 100)
    table = branch_x_pu100(n)
    assert table.loc["T132_33", "x_pu100"] == pytest.approx(0.25)
    assert table.loc["L400", "x_pu100"] == pytest.approx(0.009831)


def test_conversion_happens_once_and_round_trips():
    n = _pu100_network()
    before = pd.concat([n.lines[["r", "x", "b"]], n.transformers[["r", "x", "b"]]])
    convert_network_pu100_to_pypsa(n, LOG)
    assert get_branch_units(n) == UNITS_PYPSA
    with pytest.raises(ValueError, match="already in PyPSA units"):
        convert_network_pu100_to_pypsa(n, LOG)
    convert_network_pypsa_to_pu100(n, LOG)
    after = pd.concat([n.lines[["r", "x", "b"]], n.transformers[["r", "x", "b"]]])
    pd.testing.assert_frame_equal(before, after)
    assert n.meta[BRANCH_UNITS_META_KEY] == UNITS_PU100


def test_conversion_refuses_missing_bus_voltage():
    n = _pu100_network()
    n.buses.loc["D1", "v_nom"] = np.nan
    with pytest.raises(ValueError, match="v_nom"):
        convert_network_pu100_to_pypsa(n, LOG)


def test_transformer_rating_change_preserves_impedance():
    n = _pu100_network()
    convert_network_pu100_to_pypsa(n, LOG)
    n.calculate_dependent_values()
    x_pu = n.transformers.x_pu.copy(); b_pu = n.transformers.b_pu.copy()
    set_transformer_s_nom_preserving_impedance(n, ["T400_132", "T132_33"], 9999.0)
    n.calculate_dependent_values()
    assert n.transformers.at["T400_132", "s_nom"] == 9999.0
    pd.testing.assert_series_equal(n.transformers.x_pu, x_pu)
    pd.testing.assert_series_equal(n.transformers.b_pu, b_pu)


def test_branch_values_for_network_follows_network_units():
    n = _pu100_network()
    values = {"r": 0.000596, "x": 0.011743, "b": 0.688588, "s_nom": 3326.0}
    assert branch_values_for_network(n, "Line", values, "A4") == values
    convert_network_pu100_to_pypsa(n, LOG)
    out = branch_values_for_network(n, "Line", values, "A4")
    assert out["x"] == pytest.approx(0.011743 * 1600)
    assert out["s_nom"] == 3326.0


# ── zero / missing impedance policy (item N1 and the base sheets) ────────────

def test_policy_keeps_published_zero_r_and_b():
    values, assumed, missing = resolve_source_impedance_pu100(0.0, 0.00125, 0.0, "line",
                                                              circuit_type="SSSC", length_km=0.0)
    assert values == {"r": 0.0, "x": 0.00125, "b": 0.0}
    assert not assumed and not missing


def test_policy_zero_x_bus_section_gets_coupler():
    for ctype, length in [("Zero Length", 0.0), ("Cable", 0.25), ("OHL", 1.0)]:
        values, assumed, _ = resolve_source_impedance_pu100(0.0, 0.0, 0.0, "line",
                                                            circuit_type=ctype, length_km=length)
        assert values["x"] == ELECTRICAL_DEFAULTS["coupler"]["x"] == 1.0e-4
        assert values["r"] == 0.0 and values["b"] == 0.0
        assert assumed == ["x (zero-reactance bus section: coupler)"]


def test_policy_zero_x_long_circuit_is_missing_or_default():
    values, _, missing = resolve_source_impedance_pu100(2.4e-5, 0.0, 0.0, "cable",
                                                        circuit_type="Cable", length_km=205.0,
                                                        fill_missing=False)
    assert missing == ["x"] and values["x"] is None
    values, _, _ = resolve_source_impedance_pu100(2.4e-5, 0.0, 0.0, "cable",
                                                  circuit_type="Cable", length_km=205.0)
    assert values["x"] == ELECTRICAL_DEFAULTS["cable"]["x"]


def test_policy_unpublished_x_is_never_a_coupler():
    # GB_network.xlsx BMU edges: no impedance and no length at all
    values, assumed, _ = resolve_source_impedance_pu100(np.nan, np.nan, np.nan, "line",
                                                        circuit_type="", length_km=0.0)
    assert values == {k: ELECTRICAL_DEFAULTS["line"][k] for k in ("r", "x", "b")}
    assert "x (type default)" in assumed


def _upgrade_network():
    n = pypsa.Network()
    for bus in ["BRNX4-", "BRNX4A", "SALH41", "SALH42", "LONG41", "LONG42", "TRAF41", "TRAF11"]:
        n.add("Bus", bus, v_nom=132.0 if bus.endswith("11") else 400.0)
    return n


def _record(node1, node2, r, x, b, rating, ctype, ohl=0.0, cable=0.0):
    return {"Node1": node1, "Node2": node2, "Year": 2029, "R (% on 100 MVA)": r,
            "X (% on 100 MVA)": x, "B (% on 100 MVA)": b, "Winter Rating (MVA)": rating,
            "Circuit Type": ctype, "OHL Length (km)": ohl, "Cable Length (km)": cable}


def test_upgrade_additions_accept_zero_r_and_couple_zero_x():
    n = _upgrade_network()
    records = [
        _record("BRNX4-", "BRNX4A", 0.0, 0.0, 0.0, 2770.0, "Cable", cable=0.25),   # EGL1 terminal
        _record("SALH41", "SALH42", 0.0, 0.125, 0.0, 1831, "SSSC"),               # series compensator
        _record("LONG41", "LONG42", 0.0, 0.0, 0.0, 2000, "Cable", cable=12.0),     # placeholder
    ]
    added, failed, failures = U.apply_circuit_additions(n, records, LOG)
    assert (added, failed) == (2, 1)
    assert failures[0]["missing_fields"] == ["zero X placeholder"]
    assert n.lines.at["BRNX4-_BRNX4A_ETYS_UPG_2029", "x"] == pytest.approx(1.0e-4)   # pu100
    assert n.lines.at["SALH41_SALH42_ETYS_UPG_2029", "r"] == 0.0
    assert n.lines.at["SALH41_SALH42_ETYS_UPG_2029", "x"] == pytest.approx(0.00125)
    assert "coupler" in records[0]["_impedance_assumption"]


def test_upgrade_transformer_addition_accepts_zero_r():
    n = _upgrade_network()
    rec = {"Node1": "TRAF41", "Node2": "TRAF11", "Year": 2029, "R (% on 100 MVA)": 0.0,
           "X (% on 100 MVA)": 8.0, "B (% on 100 MVA)": 0.0, "Winter Rating (MVA)": 240}
    added, failed, _ = U.apply_transformer_additions(n, [rec], LOG)
    assert (added, failed) == (1, 0)
    assert n.transformers.iloc[0]["x"] == pytest.approx(0.08)


# ── CP30 accelerated components (item N2) ────────────────────────────────────

class _LayerNetwork:
    def __init__(self, units=None):
        self.buses = pd.DataFrame({"v_nom": [400.0] * 5, "x": np.arange(5.0), "y": [0.0] * 5},
                                  index=["BRFO41", "NORM41", "TILB41", "RICH41", "SIZE41"])
        self.lines = pd.DataFrame()
        self.links = pd.DataFrame()
        self.meta = {} if units is None else {BRANCH_UNITS_META_KEY: units}

    def add(self, component, name, **attributes):
        table = {"Bus": self.buses, "Line": self.lines, "Link": self.links}[component]
        for column in attributes:
            if column not in table.columns:
                table[column] = pd.NA
        table.loc[name, list(attributes)] = list(attributes.values())


CP30_FILE = "data/network/ETYS/cp30_n1_accelerated_components.csv"


def test_cp30_components_are_percent_and_connect_to_etys_busbars():
    n = _LayerNetwork(UNITS_PU100)
    projects = apply_cp30_reinforcement_layer(n, {"enabled": True, "components_file": CP30_FILE},
                                              2030, LOG)
    assert projects == {"AENC", "ATNC", "SCD1"}
    assert "NORM42" not in n.buses.index and "BRFO42" not in n.buses.index
    aenc = n.lines.loc[n.lines.index.str.startswith("CP30_AENC")]
    assert set(map(frozenset, zip(aenc.bus0, aenc.bus1))) == {frozenset({"BRFO41", "NORM41"})}
    # 1.1743 % on 100 MVA -> 0.011743 pu100 (~65 km of 400 kV OHL)
    assert aenc["x"].astype(float).tolist() == pytest.approx([0.011743, 0.011743])
    atnc = n.lines.loc[n.lines.index.str.startswith("CP30_ATNC")]
    assert set(atnc.bus0) | set(atnc.bus1) == {"BRFO41", "EACS41", "TILN41", "TILB41"}


def test_cp30_components_convert_when_network_already_native():
    n = _LayerNetwork(UNITS_PYPSA)
    apply_cp30_reinforcement_layer(n, {"enabled": True, "components_file": CP30_FILE}, 2030, LOG)
    assert float(n.lines.at["CP30_AENC_BRFO41_NORM42", "x"]) == pytest.approx(0.011743 * 400**2 / 100)


def test_cp30_line_without_units_is_rejected(tmp_path):
    frame = pd.read_csv(CP30_FILE)
    frame = frame.drop(columns=["impedance_units"])
    path = tmp_path / "components.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="impedance_units"):
        apply_cp30_reinforcement_layer(_LayerNetwork(), {"enabled": True, "components_file": str(path)},
                                       2030, LOG)


# ── BM voltage floor: transformers into sub-220 kV relaxed with their lines ──

def _native_network_with_voltage_column():
    n = _pu100_network()
    convert_network_pu100_to_pypsa(n, LOG)
    n.lines["voltage_kv"] = n.lines.bus0.map(n.buses.v_nom)
    return n


def test_voltage_floor_selects_lines_and_transformers_into_sub_220kv():
    n = _native_network_with_voltage_column()
    lines, trafos = bm_voltage_floor_branches(n, 220)
    assert set(lines) == {"L132", "L400_132"}        # 400 kV line kept; 400->132 edge relaxed
    assert set(trafos) == {"T400_132", "T132_33"}    # 400/275 kept


def test_apply_line_rating_overrides_relaxes_transformers_preserving_impedance():
    n = _native_network_with_voltage_column()
    n.calculate_dependent_values()
    x_pu = n.transformers.x_pu.copy()
    cfg = {"network": {"transmission": {"min_bm_constraint_voltage_kv": 220,
                                        "line_rating_overrides": {"T400_275": 1200}}}}
    apply_line_rating_overrides(n, cfg, LOG)
    assert n.lines.at["L400", "s_nom"] == 2078
    assert n.lines.at["L132", "s_nom"] == 9999.0
    assert n.transformers.at["T400_132", "s_nom"] == 9999.0
    assert n.transformers.at["T132_33", "s_nom"] == 9999.0
    assert n.transformers.at["T400_275", "s_nom"] == 1200.0
    n.calculate_dependent_values()
    pd.testing.assert_series_equal(n.transformers.x_pu, x_pu)


def test_voltage_floor_skipped_without_line_voltage_column():
    n = _pu100_network()
    convert_network_pu100_to_pypsa(n, LOG)
    apply_line_rating_overrides(n, {"network": {"transmission": {"min_bm_constraint_voltage_kv": 220,
                                                                 "line_rating_overrides": {}}}}, LOG)
    assert n.transformers.at["T132_33", "s_nom"] == 60


def test_numerical_clamp_uses_per_unit_not_raw_reactance():
    n = _native_network_with_voltage_column()
    set_transformer_s_nom_preserving_impedance(n, ["T132_33"], 9999.0)   # x = 25 pu on 9,999 MVA
    n.add("Transformer", "T_bad", bus0="D1", bus1="F3", x=20.0 * 50 / 100, s_nom=50, model="pi")  # 20 pu100
    n.calculate_dependent_values()
    x_pu = n.transformers.x_pu.copy()
    improve_numerical_conditioning(n, LOG)
    n.calculate_dependent_values()
    assert n.transformers.at["T132_33", "x_pu"] == pytest.approx(x_pu["T132_33"])   # untouched
    assert 100 * n.transformers.at["T_bad", "x_pu"] == pytest.approx(10.0)          # clamped to 10 pu100


def test_create_network_converts_every_branch_once():
    from scripts.network_build.ETYS_network import create_network
    comps = pd.DataFrame(
        [["line", "AC", "AAAA41", "BBBB41", 0.000778, 0.009831, 0.279792, 2078.0, 48.8],
         ["line", "AC", "BBBB41", "CCCC41", 0.0, 1.0e-4, 0.0, 9999.0, 0.0],
         ["transformer", "AC", "BBBB41", "BBBB11", 0.001658, 0.084025, -0.015604, 275.0, 0.0]],
        columns=["component", "carrier", "bus0", "bus1", "r", "x", "b", "s_nom", "length_km"],
        index=["L1", "COUPLER", "T1"])
    buses = pd.DataFrame({"v_nom": [400.0, 400.0, 400.0, 132.0], "carrier": "AC",
                          "lon": [-1.0, -1.1, -1.2, -1.1], "lat": [52.0, 52.1, 52.2, 52.1]},
                         index=["AAAA41", "BBBB41", "CCCC41", "BBBB11"])
    n = create_network(comps, buses, LOG, etys_upgrades_enabled=False)
    assert get_branch_units(n) == UNITS_PYPSA
    n.calculate_dependent_values()
    assert 100 * n.lines.at["L1", "x_pu"] == pytest.approx(0.009831)
    assert 100 * n.lines.at["COUPLER", "x_pu"] == pytest.approx(1.0e-4)
    assert 100 * n.transformers.at["T1", "x_pu"] == pytest.approx(0.084025)
    assert n.lines.at["L1", "x"] == pytest.approx(15.7296)
