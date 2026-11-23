"""ETYS seasonal circuit ratings and the Scottish-transmission BM floor (27 Sep 2026)."""

import logging

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.network_build.branch_attributes import (
    DEFAULT_SEASON_MONTHS, finalize_branch_attributes, is_scottish_branch, month_to_season,
    scotland_by_geography, seasonal_rating_factors, seasonal_ratings_from_record,
)
from scripts.solve.solve_network import (
    apply_line_rating_overrides, apply_seasonal_line_ratings, bm_voltage_floor_branches,
)
from config.config_loader import load_defaults

LOG = logging.getLogger(__name__)


def _network():
    """Scottish and English 400/132 kV corners, OSGB36 coordinates."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", "2030-12-31 23:00", freq="h"))
    buses = {  # name: (v_nom, x, y)
        "SCOT41": (400, 280000, 700000), "SCOT11": (132, 281000, 700000), "SCOT12": (132, 282000, 701000),
        "SCOT31": (33, 281000, 699000),
        "ENGL41": (400, 450000, 300000), "ENGL11": (132, 451000, 300000), "ENGL12": (132, 452000, 301000),
    }
    for b, (v, x, y) in buses.items():
        n.add("Bus", b, v_nom=v, x=x, y=y)
    lines = [  # name, bus0, bus1, winter, spring, summer, autumn, licensee
        ("S400", "SCOT41", "ENGL41", 2000, 1900, 1740, 1900, "SPT"),
        ("S132", "SCOT11", "SCOT12", 200, 190, 175, 190, "SHE"),
        ("S33", "SCOT31", "SCOT31", 40, np.nan, np.nan, np.nan, "SHE"),
        ("E132", "ENGL11", "ENGL12", 200, 190, 175, 190, "NGET"),
        ("E400", "ENGL41", "ENGL41", 3000, np.nan, np.nan, np.nan, None),   # e.g. an OFTO / BMU edge
        ("COUP", "ENGL11", "ENGL12", 9999, 9999, 9999, 9999, "NGET"),
    ]
    for name, b0, b1, w, sp, su, au, lic in lines:
        n.add("Line", name, bus0=b0, bus1=b1, x=1.0, s_nom=w, s_nom_spring=sp, s_nom_summer=su,
              s_nom_autumn=au, etys_licensee=lic)
    n.lines["voltage_kv"] = n.lines.bus0.map(n.buses.v_nom)
    n.add("Transformer", "ST", bus0="SCOT41", bus1="SCOT11", x=0.2, s_nom=240, etys_licensee="SHE")
    n.add("Transformer", "ST33", bus0="SCOT11", bus1="SCOT31", x=0.2, s_nom=60, etys_licensee="SHE")
    n.add("Transformer", "ET", bus0="ENGL41", bus1="ENGL11", x=0.2, s_nom=240, etys_licensee="NGET")
    return n


def test_seasonal_ratings_from_appendix_b_record():
    rec = {"Winter Rating (MVA)": 2078, "Spring Rating (MVA)": 2078,
           "Summer Rating (MVA)": 1834, "Autumn Rating (MVA)": "TBC"}
    out = seasonal_ratings_from_record(rec)
    assert out["s_nom_summer"] == 1834 and out["s_nom_spring"] == 2078
    assert np.isnan(out["s_nom_autumn"])


def test_finalize_fills_missing_by_class_median_and_keeps_relaxed():
    n = _network()
    finalize_branch_attributes(n, LOG)
    L = n.lines
    assert L.at["S400", "seasonal_rating_source"] == "etys"
    assert L.at["COUP", "seasonal_rating_source"] == "relaxed" and L.at["COUP", "s_nom_summer"] == 9999
    # E400 (400 kV, no seasonal values) takes the 400 kV class median (S400 only: 0.87)
    assert L.at["E400", "seasonal_rating_source"] == "class_median_ratio"
    assert L.at["E400", "s_nom_summer"] == pytest.approx(3000 * 1740 / 2000)
    assert (n.transformers.seasonal_rating_source == "winter_only").all()
    assert (n.transformers.s_nom_summer == n.transformers.s_nom).all()
    assert L.at["E400", "etys_licensee"] == "NGET"          # site majority (ENGL)
    assert L.at["E400", "etys_licensee_source"] == "site_majority"


def test_season_months_cover_the_year_once():
    m = month_to_season(DEFAULT_SEASON_MONTHS)
    assert sorted(m) == list(range(1, 13))
    # ER P27 / ACE 104 seasons (Northern Powergrid IMP/001/011 Table 1)
    assert [m[k] for k in (11, 12, 1, 2)] == ["winter"] * 4
    assert m[3] == m[4] == "spring" and m[9] == m[10] == "autumn"
    assert [m[k] for k in (5, 6, 7, 8)] == ["summer"] * 4
    with pytest.raises(ValueError):
        month_to_season({"winter": [1, 2], "summer": [2]})


def test_seasonal_factors_by_month_and_relaxation():
    n = _network(); finalize_branch_attributes(n, LOG)
    n.lines.loc["S132", "s_nom"] = 9999.0          # relaxed by the floor at solve time
    f = seasonal_rating_factors(n, n.snapshots)
    assert "S132" not in f.columns and "COUP" not in f.columns
    assert f.loc["2030-07-15 12:00", "S400"] == pytest.approx(0.87)
    assert f.loc["2030-04-15 12:00", "S400"] == pytest.approx(0.95)
    assert f.loc["2030-01-15 12:00", "S400"] == pytest.approx(1.0)
    assert f.loc["2030-03-15 12:00", "S400"] == pytest.approx(0.95)    # March is spring
    assert f.loc["2030-11-15 12:00", "S400"] == pytest.approx(1.0)     # November is winter


def test_apply_seasonal_multiplies_existing_s_max_pu():
    n = _network(); finalize_branch_attributes(n, LOG)
    n.lines_t.s_max_pu = pd.DataFrame({"S400": 0.5}, index=n.snapshots)   # e.g. an outage
    applied = apply_seasonal_line_ratings(n, {"network": {"transmission": {"seasonal_ratings": {"enabled": True}}}}, LOG)
    assert applied >= 2
    s = n.lines_t.s_max_pu
    assert s.loc["2030-07-15 12:00", "S400"] == pytest.approx(0.5 * 0.87)
    assert s.loc["2030-07-15 12:00", "E132"] == pytest.approx(175 / 200)
    assert "COUP" not in s.columns


def test_apply_seasonal_disabled_and_legacy_network():
    n = _network(); finalize_branch_attributes(n, LOG)
    assert apply_seasonal_line_ratings(n, {"network": {"transmission": {"seasonal_ratings": {"enabled": False}}}}, LOG) == 0
    legacy = _network()
    assert apply_seasonal_line_ratings(legacy, {"network": {"transmission": {}}}, LOG) == 0


def test_scottish_branches_by_licensee_and_geography():
    n = _network(); finalize_branch_attributes(n, LOG)
    scot = is_scottish_branch(n, n.lines)
    assert scot["S132"] and scot["S400"] and not scot["E132"]
    geo = scotland_by_geography(n.buses)
    assert geo["SCOT41"] and not geo["ENGL41"]
    # Galloway (Glenluce, Tongland) and Gretna are Scotland; Carlisle/Harker England
    pts = pd.DataFrame({"x": [220584, 270034, 331836, 337276, 340000, 399500],
                        "y": [557127, 552962, 567276, 561712, 556000, 653000]},
                       index=["GLLU", "TONG", "GRNA", "HAKB", "CARL", "BERWICK_TOWN"])
    g = scotland_by_geography(pts)
    assert g[["GLLU", "TONG", "GRNA"]].all() and not g[["HAKB", "CARL", "BERWICK_TOWN"]].any()


def test_floor_keeps_scottish_132kv_constrained_but_relaxes_ew_and_collectors():
    n = _network(); finalize_branch_attributes(n, LOG)
    lines, trafos = bm_voltage_floor_branches(n, 220)          # old E&W-style rule everywhere
    assert {"S132", "S33", "E132"} <= set(lines)
    lines, trafos, kl, kt = bm_voltage_floor_branches(n, 220, True, 132, return_scottish=True)
    assert "S132" in kl and "S132" not in lines
    assert "S33" in lines and "E132" in lines              # collectors and DNO stay relaxed
    assert "ST" in kt and "ST33" in trafos and "ET" in trafos
    cfg = {"network": {"transmission": {"min_bm_constraint_voltage_kv": 220, "line_rating_overrides": {},
                                        "bm_voltage_floor": {"scottish_transmission_constrained": True,
                                                             "scottish_min_constrained_kv": 132}}}}
    apply_line_rating_overrides(n, cfg, LOG)
    assert n.lines.at["S132", "s_nom"] == 200 and n.lines.at["E132", "s_nom"] == 9999
    assert n.transformers.at["ST", "s_nom"] == 240 and n.transformers.at["ET", "s_nom"] == 9999


def test_floor_scottish_switch_off_restores_old_rule():
    n = _network(); finalize_branch_attributes(n, LOG)
    cfg = {"network": {"transmission": {"min_bm_constraint_voltage_kv": 220, "line_rating_overrides": {},
                                        "bm_voltage_floor": {"scottish_transmission_constrained": False}}}}
    apply_line_rating_overrides(n, cfg, LOG)
    assert n.lines.at["S132", "s_nom"] == 9999


def test_cp30_components_carry_seasonal_ratings_and_licensee():
    comps = pd.read_csv("data/network/ETYS/cp30_n1_accelerated_components.csv")
    lines = comps[comps.component == "line"]
    assert lines[["s_nom_spring", "s_nom_summer", "s_nom_autumn"]].notna().all().all()
    assert (lines.etys_licensee == "NGET").all()
    tilb = lines[lines.name == "CP30_ATNC_TILB41_TILN41_1"].iloc[0]
    assert tilb.s_nom_summer == 2591 and tilb.s_nom == 2827


def test_defaults_yaml_season_months_match_code_default():
    import yaml
    cfg = load_defaults()
    months = cfg["network"]["transmission"]["seasonal_ratings"]["season_months"]
    assert {k: sorted(v) for k, v in months.items()} == {k: sorted(v) for k, v in DEFAULT_SEASON_MONTHS.items()}


def test_seasonal_factor_is_separate_from_secured_boundary_and_skips_neso_uniform():
    n = _network(); finalize_branch_attributes(n, LOG)
    # secured_capability: boundary sums are a separate constraint; line factors still apply
    cfg = {"network": {"transmission": {"outage_schedule": {"enabled": True, "source": "secured_capability"}}}}
    assert apply_seasonal_line_ratings(n, cfg, LOG) >= 2
    # legacy NESO uniform_s_max_pu limits already carry the season: not re-scaled
    n = _network(); finalize_branch_attributes(n, LOG)
    n.lines_t.s_max_pu = pd.DataFrame({"S400": 0.6}, index=n.snapshots)
    cfg = {"network": {"transmission": {"outage_schedule": {"enabled": True, "source": "neso",
                                                            "neso": {"constraint_mode": "uniform_s_max_pu"}}}}}
    apply_seasonal_line_ratings(n, cfg, LOG)
    assert n.lines_t.s_max_pu.loc["2030-07-15 12:00", "S400"] == pytest.approx(0.6)
    assert n.lines_t.s_max_pu.loc["2030-07-15 12:00", "E132"] == pytest.approx(175 / 200)
