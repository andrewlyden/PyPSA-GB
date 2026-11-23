"""Network-build fixes of 15 Oct 2026 (CP30 2030 network audit).

* CP30-dated Appendix B works: later-snapshot rows CP30 N1 requires by 2030
  (data/network/ETYS/cp30_appendix_b_dated_works.csv, ETYS_upgrades.add_cp30_dated_works).
* Appendix B and base rows repeated with swapped terminals are built once.
* Connah's Quay 2027 terminal reconciliation (CONQ42 is the base busbar CONQ41).
* A deferred HVDC link stays out even when a pulled-forward work creates a bus
  its terminal could resolve to (Western Isles vs Fanellan).
* Per-bus coordinate overrides; site positions for buses first built by
  2032-33 rows; the E4D3 manifest terminal.
* Bus sections and couplers as one node in bus_branch_capacity and weak-bus relief.

The general fixes are plain behaviour for every build (user decision, 15 Oct
2026; no switches). The Connah's Quay remap is a documented topology correction,
named true in config/defaults/base/network.yaml beside the others. The CP30-dated Appendix B rows
are a CP30 delivery choice, set once in config/defaults/method/brownfield.yaml
(network.reinforcements.appendix_b_dated_works_file).
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.network_build import ETYS_upgrades as EU  # noqa: E402
from scripts.network_build.ETYS_network import (  # noqa: E402
    apply_bus_coordinate_overrides,
    drop_reversed_duplicate_branches,
)
from scripts.network_build.branch_units import UNITS_PU100, set_branch_units  # noqa: E402
from config.config_loader import load_defaults

LOG = logging.getLogger(__name__)
DATED = ROOT / "data/network/ETYS/cp30_appendix_b_dated_works.csv"
MANIFEST = ROOT / "data/network/ETYS/cp30_n1_reinforcement_manifest.csv"
APPENDIX_B = ROOT / "data/network/ETYS/ETYS 2024 Appendix-B V1.xlsx"


def _row(n1, n2, year, status="Addition", r=0.01, x=0.1, b=1.0, rating=2000.0, operator="SHE"):
    return {"Node1": n1, "Node2": n2, "Year": year, "Status": status, "operator": operator,
            "R (% on 100 MVA)": r, "X (% on 100 MVA)": x, "B (% on 100 MVA)": b,
            "Winter Rating (MVA)": rating, "OHL Length (km)": 10.0, "Cable Length (km)": 0.0,
            "Circuit Type": "OHL"}


def _work(n1, n2, appendix_year, operation="addition", occurrence=1, cp30_year=2030, **values):
    return {"project_code": "TEST", "in_service_year": cp30_year, "component": "circuit",
            "operation": operation, "etys_sheet": "B-2-2a", "operator": "SHE", "node1": n1,
            "node2": n2, "appendix_b_year": appendix_year, "occurrence": occurrence,
            "r": values.get("r", 0.01), "x": values.get("x", 0.1), "b": values.get("b", 1.0),
            "impedance_units": "percent_on_100MVA", "winter_rating_mva": values.get("rating", 2000.0),
            "source": "test source"}


def _empty_categorized():
    return {"circuits": {"additions": [], "removals": [], "changes": []},
            "transformers": {"additions": [], "removals": [], "changes": []},
            "hvdc": {"additions": []}}


# ── reversed duplicates ─────────────────────────────────────────────────────

def test_reversed_duplicate_appendix_b_rows_are_dropped_but_parallels_kept():
    rows = [_row("FETT4Q", "KINT4J", 2027), _row("KINT4J", "FETT4Q", 2027),
            _row("GRNS4L", "NEDE4J", 2032), _row("GRNS4L", "NEDE4J", 2032),
            _row("AAAA4-", "BBBB4-", 2027), _row("BBBB4-", "AAAA4-", 2027, rating=1000.0)]
    kept = EU.drop_reversed_duplicate_records(rows, "test", LOG)
    pairs = [(r["Node1"], r["Node2"], r["Winter Rating (MVA)"]) for r in kept]
    assert pairs.count(("FETT4Q", "KINT4J", 2000.0)) == 1
    assert ("KINT4J", "FETT4Q", 2000.0) not in pairs
    assert pairs.count(("GRNS4L", "NEDE4J", 2000.0)) == 2       # same orientation: real parallels
    assert ("BBBB4-", "AAAA4-", 1000.0) in pairs                 # different rating: not a repeat


def test_filter_upgrades_by_year_drops_reversed_repeats():
    circuits = pd.DataFrame([_row("FETT4Q", "KINT4J", 2027), _row("KINT4J", "FETT4Q", 2027)])
    data = {"circuits": circuits, "transformers": pd.DataFrame(), "hvdc": pd.DataFrame()}
    assert len(EU.filter_upgrades_by_year(data, 2030, LOG)["circuits"]["additions"]) == 1


def test_network_fixes_have_no_switches_and_connahs_quay_is_a_documented_correction():
    import inspect

    import yaml
    assert not hasattr(EU, "NETWORK_FIX_DEFAULTS") and not hasattr(EU, "resolve_network_fixes")
    assert "network_fixes" not in inspect.signature(EU.apply_etys_network_upgrades).parameters
    assert inspect.signature(EU.add_missing_buses_from_upgrades).parameters["max_iterations"].default == 25
    defaults = load_defaults()
    assert "network_fixes" not in defaults["network"]["upgrades"]
    # Connah's Quay: named true in the defaults' network.upgrades mapping, like the other documented corrections
    defaults_mapping = defaults["network"]["upgrades"]["documented_topology_corrections"]
    assert defaults_mapping["connahs_quay_2027_terminal_reconciliation"] is True
    assert set(defaults_mapping) == EU.DOCUMENTED_TOPOLOGY_CORRECTION_IDS
    assert "connahs_quay_2027_terminal_reconciliation" in EU.enabled_documented_topology_corrections(defaults_mapping)
    assert "connahs_quay_2027_terminal_reconciliation" not in EU.enabled_documented_topology_corrections(
        {"connahs_quay_2027_terminal_reconciliation": False})


# ── CP30-dated works ────────────────────────────────────────────────────────

def _upgrades(rows):
    return {"circuits": pd.DataFrame(rows), "transformers": pd.DataFrame(), "hvdc": pd.DataFrame()}


def test_dated_work_is_pulled_with_the_workbook_record():
    data = _upgrades([_row("BANN4J", "CAAI4J", 2032), _row("OTHR4-", "OTHR4A", 2032)])
    categorized = _empty_categorized()
    works = pd.DataFrame([_work("BANN4J", "CAAI4J", 2032)])
    pulled = EU.add_cp30_dated_works(data, categorized, 2030, works, LOG)
    assert len(pulled) == 1
    added = categorized["circuits"]["additions"]
    assert [(r["Node1"], r["Node2"]) for r in added] == [("BANN4J", "CAAI4J")]
    assert added[0]["_cp30_dated_work"] == "TEST"


def test_dated_work_not_pulled_before_its_cp30_year_or_once_in_the_snapshot():
    data = _upgrades([_row("BANN4J", "CAAI4J", 2032)])
    works = pd.DataFrame([_work("BANN4J", "CAAI4J", 2032, cp30_year=2031)])
    categorized = _empty_categorized()
    assert EU.add_cp30_dated_works(data, categorized, 2030, works, LOG) == []
    works = pd.DataFrame([_work("BANN4J", "CAAI4J", 2032)])
    categorized = _empty_categorized()
    assert EU.add_cp30_dated_works(data, categorized, 2035, works, LOG) == []   # ordinary filter has it


def test_dated_work_fails_loudly_when_the_workbook_disagrees():
    data = _upgrades([_row("BANN4J", "CAAI4J", 2032, rating=2780.0)])
    works = pd.DataFrame([_work("BANN4J", "CAAI4J", 2032, rating=2000.0)])
    with pytest.raises(ValueError, match="Winter Rating"):
        EU.add_cp30_dated_works(data, _empty_categorized(), 2030, works, LOG)
    works = pd.DataFrame([_work("BANN4J", "XXXX4J", 2032)])
    with pytest.raises(ValueError, match="not in the Appendix B workbook"):
        EU.add_cp30_dated_works(data, _empty_categorized(), 2030, works, LOG)


def test_dated_work_occurrence_selects_repeated_published_rows():
    data = _upgrades([_row("GRNS4L", "NEDE4J", 2032), _row("GRNS4L", "NEDE4J", 2032)])
    works = pd.DataFrame([_work("GRNS4L", "NEDE4J", 2032, occurrence=1),
                          _work("GRNS4L", "NEDE4J", 2032, occurrence=2)])
    categorized = _empty_categorized()
    assert len(EU.add_cp30_dated_works(data, categorized, 2030, works, LOG)) == 2
    # already in scope (e.g. pulled as HVDC enabling works) -> not duplicated
    assert EU.add_cp30_dated_works(data, categorized, 2030, works, LOG) == []
    assert len(categorized["circuits"]["additions"]) == 2


def test_dated_works_table_rejects_undeclared_units(tmp_path):
    bad = pd.DataFrame([_work("A", "B", 2032)])
    bad["impedance_units"] = "ohm"
    path = tmp_path / "w.csv"
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError, match="percent_on_100MVA"):
        EU.load_cp30_dated_works(path)


def test_failed_dated_work_raises():
    pulled = [{**_row("BANN4J", "CAAI4J", 2032), "_cp30_dated_work": "SLU4", "_component": "circuits"}]
    failures = {"circuits": [{"type": "addition", "reason": "bus_not_found",
                              "node1": "BANN4J", "node2": "CAAI4J"}], "transformers": [], "hvdc": []}
    with pytest.raises(ValueError, match="SLU4"):
        EU.check_cp30_dated_works_applied(pulled, failures)
    EU.check_cp30_dated_works_applied(pulled, {"circuits": [], "transformers": [], "hvdc": []})


# ── the real CP30 table against the real workbook ──────────────────────────

def test_dated_works_table_is_sourced_and_due_by_2030():
    works = EU.load_cp30_dated_works(DATED)
    assert (works["in_service_year"] <= 2030).all()
    assert (works["appendix_b_year"] > 2030).all()
    assert works["source"].str.contains("CP30 Data Workbook V003 sheet N1").all()
    assert works["source"].str.contains("ETYS 2024 Appendix B").all()
    expected = {"SLU4", "BLN4", "BBNC", "BPNC", "NNNC", "TKUP", "DWNO", "VERE", "EHRE", "LNRE",
                "EBRE", "GCHW", "PCR1", "IFR1", "PTNO", "DFRE", "ETRE"}
    assert set(works["project_code"]) == expected


@pytest.mark.skipif(not APPENDIX_B.is_file(), reason="ETYS 2024 Appendix B workbook not present")
def test_every_dated_work_matches_the_appendix_b_workbook():
    data = EU.load_etys_upgrade_data(str(APPENDIX_B), LOG)
    categorized = EU.filter_upgrades_by_year(data, 2030, LOG)
    works = EU.load_cp30_dated_works(DATED)
    pulled = EU.add_cp30_dated_works(data, categorized, 2030, works, LOG)
    assert len(pulled) == len(works)        # every row found, values equal (else it raises)


def test_manifest_covers_every_dated_work_and_e4d3_names_the_built_busbar():
    manifest = pd.read_csv(MANIFEST).set_index("project_code")
    works = pd.read_csv(DATED)
    for code in set(works["project_code"]) | {"PTC1"}:
        assert manifest.at[code, "representation_status"] == "represented_in_etys"
        assert str(manifest.at[code, "expected_component"]).strip()
    assert manifest.at["E4D3", "expected_bus1"] == "PEHE4J"


def test_reinforcement_audit_checks_dated_works_against_the_network(tmp_path):
    from scripts.CP30.reinforcements import audit_cp30_reinforcements

    reference = pd.DataFrame([{
        "source_sheet": "N1", "Code": "IFR1", "Description": "Feckenham-Ironbridge",
        "Latest Delivery Year (updated September 2024)": 2028, "Required for CP30": "Y",
    }])
    ref_path = tmp_path / "ref.csv"
    reference.to_csv(ref_path, index=False)
    manifest = pd.DataFrame([{
        "project_code": "IFR1", "representation_status": "represented_in_etys",
        "evidence": "e", "expected_component": "line", "expected_bus0": "FECK41",
        "expected_bus1": "IRON41", "expected_min_nominal": "3100", "mapping_basis": "m",
        "evidence_status": "reviewed_public_branch_mapping", "directional_effect_if_missing": "x",
    }])
    man_path = tmp_path / "man.csv"
    manifest.to_csv(man_path, index=False)

    class Net:
        links = pd.DataFrame(columns=["bus0", "bus1", "p_nom"])
        transformers = pd.DataFrame(columns=["bus0", "bus1", "s_nom"])

        def __init__(self, rating):
            self.lines = pd.DataFrame([{"bus0": "FECK41", "bus1": "IRON41", "s_nom": rating}])

    ok = audit_cp30_reinforcements(ref_path, man_path, 2030, tmp_path / "a.csv", LOG, network=Net(3100.0))
    assert ok.set_index("project_code").at["IFR1", "audit_status"] == "represented_in_etys"
    bad = audit_cp30_reinforcements(ref_path, man_path, 2030, tmp_path / "b.csv", LOG, network=Net(2009.0))
    assert bad.set_index("project_code").at["IFR1", "audit_status"] == "expected_etys_component_missing"


# ── Connah's Quay reconciliation ────────────────────────────────────────────

def _conq_network():
    n = pypsa.Network()
    for bus, x in (("BODE41", 0.0), ("CONQ41", 30000.0), ("GWYN4A", -30000.0)):
        n.add("Bus", bus, v_nom=400.0, x=x, y=0.0)
    n.add("Line", "CONQ41_GWYN4A_0", bus0="CONQ41", bus1="GWYN4A", r=0.0003, x=0.005, s_nom=2665.0)
    n.add("Line", "BODE41_GWYN4A_0", bus0="BODE41", bus1="GWYN4A", r=0.0003, x=0.005, s_nom=1056.0)
    return n


def test_connahs_quay_removal_and_addition_reconciled_unless_switched_off():
    off = {"connahs_quay_2027_terminal_reconciliation": False}
    removal = _row("CONQ42", "GWYN4A", 2027, status="Removed", rating=2665.0, operator="NGET")
    n = _conq_network()
    ok, failed, _ = EU.apply_circuit_removals(n, [dict(removal)], LOG)
    assert (ok, failed) == (1, 0) and "CONQ41_GWYN4A_0" not in n.lines.index
    n = _conq_network()
    ok, failed, _ = EU.apply_circuit_removals(n, [dict(removal)], LOG, off)
    assert (ok, failed) == (0, 1) and "CONQ41_GWYN4A_0" in n.lines.index

    for mapping, expected in ((None, "CONQ41"), ({}, "CONQ41"), (off, "CONQ42")):
        categorized = _empty_categorized()
        categorized["circuits"]["additions"].append(_row("BODE41", "CONQ42", 2027, rating=2665.0, operator="NGET"))
        EU.apply_addition_node_remaps(categorized, mapping, LOG)
        rec = categorized["circuits"]["additions"][0]
        assert (rec["Node1"], rec["Node2"]) == ("BODE41", expected)


# ── Glenglass GLGL1B reconciliation (F25.12, 27 Oct 2026) ──────────────────

def _glgl_network():
    n = pypsa.Network()
    set_branch_units(n, UNITS_PU100)
    for bus, v, x in (("GLGL1Q", 132.0, 0.0), ("GLGL1R", 132.0, 0.0), ("GLGL3-", 33.0, 0.0),
                      ("SAKN1-", 132.0, -16000.0), ("SANQ1-", 132.0, 4000.0), ("BLAH1-", 132.0, -12000.0),
                      ("GHPS1-", 132.0, 9000.0)):
        n.add("Bus", bus, v_nom=v, x=x, y=0.0, carrier="AC")
    n.add("Line", "GLGL1Q_SAKN1-_0", bus0="GLGL1Q", bus1="SAKN1-", r=0.002541, x=0.005842, b=0.110847,
          s_nom=166.0, carrier="AC")
    for bar in ("GLGL1Q", "GLGL1R"):
        n.add("Line", f"BLAH1-_{bar}_0", bus0="BLAH1-", bus1=bar, r=0.0048, x=0.027, b=0.011, s_nom=297.0)
        n.add("Transformer", f"{bar}_GLGL3-_0", bus0=bar, bus1="GLGL3-", r=0.0033, x=0.28, s_nom=90.0)
    n.add("Load", "L", bus="GLGL3-", p_set=1.0)
    n.add("Load", "L2", bus="SANQ1-", p_set=1.0)
    n.add("Load", "L3", bus="GHPS1-", p_set=1.0)
    return n


def _glgl_rows():
    return pd.DataFrame([
        _row("GLGL1-", "SANQ1-", 2025, r=0.0629, x=0.3444, b=6.5906, rating=283.0, operator="SPT"),
        _row("GHPS1-", "GLGL1-", 2027, r=0.2554, x=1.7761, b=0.6376, rating=12.0, operator="SPT"),
        _row("GLGL1B", "SAKN1-", 2029, status="Change", r=0.2541, x=0.5842, b=11.0847, rating=181.0, operator="SPT"),
        _row("GLGL1B", "SANQ1-", 2029, status="Change", r=0.0629, x=0.3444, b=6.5906, rating=241.0, operator="SPT"),
        _row("GLGL1B", "SAKN1-", 2033, status="Change", r=0.2541, x=0.5842, b=11.0847, rating=166.0, operator="SPT"),
        _row("GLGL1B", "SANQ1-", 2033, status="Change", r=0.0629, x=0.3444, b=6.5906, rating=283.0, operator="SPT"),
    ])


@pytest.mark.parametrize("year, sakn, sanq", [(2030, 181.0, 241.0), (2035, 166.0, 283.0), (2027, 166.0, 283.0)])
def test_glenglass_change_rows_apply_and_the_sanquhar_section_is_coupled_to_glgl1q(monkeypatch, tmp_path,
                                                                                  year, sakn, sanq):
    from scripts.utilities.spatial_utils import bus_section_groups
    monkeypatch.setattr(EU, "load_etys_upgrade_data", lambda *a, **k: {
        "circuits": _glgl_rows(), "transformers": pd.DataFrame(), "hvdc": pd.DataFrame()})
    n = _glgl_network()
    audit = tmp_path / "audit.csv"
    EU.apply_etys_network_upgrades(n, year, etys_file="x.xlsx", audit_path=str(audit), logger=LOG)
    lines = n.lines
    assert "GLGL1B" not in n.buses.index
    coupler = lines.loc["GLGL1-_GLGL1Q_ETYS_2025_BUS_COUPLER"]
    assert (coupler.bus0, coupler.bus1, coupler.s_nom, coupler.x) == ("GLGL1-", "GLGL1Q", 9999.0, 1.0e-4)
    groups = bus_section_groups(n)
    assert groups["GLGL1-"] == groups["GLGL1Q"] != groups["GLGL1R"]          # one node with GLGL1Q
    assert lines.at["GLGL1Q_SAKN1-_0", "s_nom"] == sakn
    cable = lines[(lines.bus0 == "GLGL1-") & (lines.bus1 == "SANQ1-")]
    assert cable["s_nom"].tolist() == [sanq]
    rows = pd.read_csv(audit)
    assert "base_topology_terminal_mismatch" not in set(rows["audit_status"])
    if year >= 2027:     # the Glenmuckloch circuits keep their names (the line_rating_overrides key on them)
        assert "GHPS1-_GLGL1-_ETYS_UPG_2027" in lines.index


def test_glenglass_reconciliation_can_be_switched_off(monkeypatch, tmp_path):
    monkeypatch.setattr(EU, "load_etys_upgrade_data", lambda *a, **k: {
        "circuits": _glgl_rows(), "transformers": pd.DataFrame(), "hvdc": pd.DataFrame()})
    n = _glgl_network()
    audit = tmp_path / "audit.csv"
    EU.apply_etys_network_upgrades(n, 2030, etys_file="x.xlsx", audit_path=str(audit), logger=LOG,
                                   documented_topology_corrections={"glenglass_2025_terminal_reconciliation": False})
    assert "GLGL1-_GLGL1Q_ETYS_2025_BUS_COUPLER" not in n.lines.index
    assert n.lines.at["GLGL1Q_SAKN1-_0", "s_nom"] == 166.0
    rows = pd.read_csv(audit)
    assert (rows["audit_status"] == "base_topology_terminal_mismatch").sum() == 2


# ── deferred HVDC link stays out ────────────────────────────────────────────

def test_deferred_hvdc_link_stays_out_when_a_dated_work_creates_its_site_bus(monkeypatch, tmp_path):
    n = pypsa.Network()
    set_branch_units(n, UNITS_PU100)
    for bus, x, y in (("BEAU4-", 0.0, 0.0), ("BING4-", 20000.0, 0.0), ("LEWI41", -90000.0, 50000.0),
                      ("LEWI11", -90000.0, 50000.0)):
        n.add("Bus", bus, v_nom=400.0 if bus[4] == "4" else 132.0, x=x, y=y, carrier="AC")
    n.add("Line", "BEAU4-_BING4-_0", bus0="BEAU4-", bus1="BING4-", r=0.0003, x=0.005, s_nom=2780.0)
    n.add("Transformer", "LEWI41_LEWI11_0", bus0="LEWI41", bus1="LEWI11", r=0.001, x=0.04, s_nom=240.0)
    n.add("Load", "L", bus="LEWI11", p_set=1.0)   # keep the Lewis pocket from being pruned
    circuits = pd.DataFrame([
        _row("BEAU4-", "FNLN4A", 2032, rating=2780.0),
        _row("FNLN4A", "FNLN4R", 2032, rating=9999.0),
        _row("LEWI4R", "LEWI4J", 2032, rating=9999.0),
    ])
    hvdc = pd.DataFrame([{"Interconnector Name": "Western Isles HVDC Link", "Node 1": "LEWI4R",
                          "Node 2": "FNLN4R", "Existing": "No", "Planned from year": 2030,
                          "Winter Rating (MVA)": 1800.0}])
    monkeypatch.setattr(EU, "load_etys_upgrade_data", lambda *a, **k: {
        "circuits": circuits, "transformers": pd.DataFrame(), "hvdc": hvdc})
    works = pd.DataFrame([_work("BEAU4-", "FNLN4A", 2032, rating=2780.0)])
    works_path = tmp_path / "works.csv"
    works.to_csv(works_path, index=False)
    audit = tmp_path / "audit.csv"
    EU.apply_etys_network_upgrades(n, 2030, etys_file="x.xlsx", audit_path=str(audit), logger=LOG,
                                   cp30_dated_works_file=str(works_path))
    assert "FNLN4A" in n.buses.index
    assert len(n.links) == 0
    rows = pd.read_csv(audit)
    hv = rows[rows["component"] == "hvdc"]
    assert hv["audit_status"].tolist() == ["deferred_terminals_later_year"]
    assert "applied_as_cp30_dated_work" in set(rows["audit_status"])


# ── coordinates ─────────────────────────────────────────────────────────────

def test_bus_coordinate_overrides(tmp_path):
    buses = pd.DataFrame({"v_nom": [275.0, 33.0], "lon": [1.02, 1.38], "lat": [52.34, 52.91]},
                         index=pd.Index(["TKNE21", "HOWW33"], name="name"))
    path = tmp_path / "o.csv"
    pd.DataFrame([
        {"bus": "TKNE21", "lat": 53.5, "lon": 0.85, "is_offshore": True, "source": "s"},
        {"bus": "NOPE1-", "lat": 50.0, "lon": 0.0, "is_offshore": False, "source": "s"},
    ]).to_csv(path, index=False)
    out, offshore = apply_bus_coordinate_overrides(buses, set(), str(path), LOG)
    assert (out.at["TKNE21", "lat"], out.at["TKNE21", "lon"]) == (53.5, 0.85)
    assert offshore == {"TKNE21"}
    assert out.at["HOWW33", "lat"] == 52.91


def test_shipped_coordinate_data_is_sourced():
    overrides = pd.read_csv(ROOT / "data/network/ETYS/bus_coordinate_overrides.csv")
    assert overrides["source"].notna().all()
    assert set(overrides["bus"]) >= {"CASS3-", "HOWW33", "TKNE21", "TKNW21"}
    assert overrides.set_index("bus").loc[["HOWW33", "TKNE21", "TKNW21"], "is_offshore"].all()
    sites = pd.read_csv(ROOT / "data/network/ETYS/substation_coordinates.csv").set_index("site_code")
    assert not sites.index.duplicated().any()
    for site in ("CAAI", "COFD", "GRNS", "EMMO", "HURL", "LOSD", "FNLN"):
        assert site in sites.index and np.isfinite(sites.at[site, "lat"]) and str(sites.at[site, "source"]).strip()


# ── rule inputs: declared only where switched on ─────────────────────────────

def _rule_function(name):
    import re
    text = (ROOT / "rules/network_build.smk").read_text(encoding="utf-8")
    match = re.search(rf"^def {name}\(.*?(?=^\S)", text, re.S | re.M)
    namespace = {}
    exec(match.group(0), namespace)
    return namespace[name]


def test_dated_works_input_only_for_cp30_blocks_that_name_it():
    inputs = _rule_function("get_cp30_dated_works_input")
    assert inputs({}) == []
    assert inputs({"network": {"reinforcements": {"enabled": False, "appendix_b_dated_works_file": "x.csv"}}}) == []
    assert inputs({"network": {"reinforcements": {"enabled": True, "manifest_file": "m.csv"}}}) == []   # FES 2050 block
    assert inputs({"network": {"reinforcements": {"enabled": True, "appendix_b_dated_works_file": "a.csv"}}}) == ["a.csv"]


# ── bus sections as one node in weak-bus relief ─────────────────────────────

def _abne_network():
    n = pypsa.Network()
    for bus, v, x in (("ABNE1Q", 132.0, 0.0), ("ABNE1R", 132.0, 10.0), ("ABNE3-", 33.0, 0.0),
                      ("GRID4-", 400.0, 5000.0), ("BIGG1-", 132.0, 8000.0), ("BIGG4-", 400.0, 8000.0)):
        n.add("Bus", bus, v_nom=v, x=x, y=0.0)
    n.add("Line", "ABNE1Q_ABNE1R_0", bus0="ABNE1Q", bus1="ABNE1R", x=0.01, r=0.0, s_nom=9999.0)
    n.add("Line", "ABNE1R_GRID_0", bus0="ABNE1R", bus1="GRID4-", x=1.0, r=0.1, s_nom=264.0)
    n.add("Line", "GRID_BIGG_0", bus0="GRID4-", bus1="BIGG1-", x=1.0, r=0.1, s_nom=2000.0)
    n.add("Transformer", "ABNE1Q_ABNE3-_0", bus0="ABNE1Q", bus1="ABNE3-", x=0.1, r=0.01, s_nom=120.0)
    return n


def test_bus_sections_count_as_one_node():
    from scripts.utilities.spatial_utils import bus_branch_capacity, bus_section_groups
    n = _abne_network()
    assert bus_section_groups(n)["ABNE1R"] == "ABNE1Q"
    new = bus_branch_capacity(n)
    # Per busbar the 9,999 MVA coupler made ABNE1Q a 10,119 MVA bus; as one node it is 264 + 120.
    assert new["ABNE1Q"] == new["ABNE1R"] == 264.0 + 120.0
    assert new["GRID4-"] == 264.0 + 2000.0                 # a bus with no coupler: its own branches


def test_weak_bus_relief_with_bus_sections_moves_the_abne_pocket():
    from scripts.utilities.spatial_utils import relieve_weak_bus_placements
    n = _abne_network()
    sites = pd.DataFrame([
        {"site_name": "CP30_LAES_1", "technology": "LAES", "capacity_mw": 322.0, "bus": "ABNE1Q",
         "fleet_origin": "", "data_source": "proxy"},
        {"site_name": "CP30_Battery_1", "technology": "Battery", "capacity_mw": 300.0, "bus": "ABNE1R",
         "fleet_origin": "", "data_source": "proxy"},
    ])
    out, moves = relieve_weak_bus_placements(sites, n, 1.5, 40.0, 132.0, log=LOG)
    # node ABNE1Q+ABNE1R: 622 MW against 1.5 x 384; the larger row leaves the node, never to its own coupler twin
    assert list(moves["site_name"]) == ["CP30_LAES_1"]
    assert out.loc[0, "bus"] not in ("ABNE1Q", "ABNE1R")
