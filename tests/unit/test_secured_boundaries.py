"""Secured boundary limits (ETYS capability x NOA seasonal factors)."""

import pandas as pd
import pypsa
import pytest

from scripts.solve.secured_boundaries import (
    DEFAULT_CAPABILITY,
    DEFAULT_GEOMETRY,
    NOA_SEASONAL_FACTORS,
    build_secured_boundary_specs,
    find_boundary_crossings,
    load_boundary_geometries,
    load_capabilities,
    seasonal_factor_series,
)


def test_capability_file_holds_the_etys_2025_values():
    cap = load_capabilities(DEFAULT_CAPABILITY, 2030, "HT")
    # ETYS 2025 boundary chart data, Capability rows, 2030
    assert cap.loc["B6", "capability_mw"] == 15354
    assert cap.loc["B4", "capability_mw"] == 7313
    assert cap.loc["EC5", "capability_mw"] == 4870
    assert set(cap["limit_type"]) <= set(NOA_SEASONAL_FACTORS)
    assert cap.loc["EC5", "limit_type"] == "voltage"


def test_default_calendar_matches_the_line_rating_seasons():
    from scripts.solve.secured_boundaries import DEFAULT_MONTH_SEASON as M
    assert [m for m, s in M.items() if s == "winter"] == [1, 2, 11, 12]
    assert [m for m, s in M.items() if s == "spring_autumn"] == [3, 4, 9, 10]
    assert [m for m, s in M.items() if s in ("summer", "summer_outage")] == [5, 6, 7, 8]


def test_noa_table_2_3_factors_by_month():
    snaps = pd.date_range("2030-01-01", "2030-12-31 23:00", freq="h")
    th = seasonal_factor_series(snaps, "thermal", NOA_SEASONAL_FACTORS,
                                {m: s for m, s in zip(range(1, 13), [
                                    "winter", "winter", "spring_autumn", "spring_autumn", "summer",
                                    "summer_outage", "summer_outage", "summer_outage", "spring_autumn",
                                    "spring_autumn", "spring_autumn", "winter"])})
    assert th["2030-12-08"].eq(1.0).all()
    assert th["2030-04-15"].eq(0.85).all()
    assert th["2030-05-15"].eq(0.80).all()
    assert th["2030-06-03"].eq(0.70).all()
    vo = seasonal_factor_series(snaps, "voltage", NOA_SEASONAL_FACTORS,
                                {m: ("summer_outage" if m in (6, 7, 8) else "winter") for m in range(1, 13)})
    assert vo["2030-06-03"].eq(0.90).all()


def _toy_network():
    """Two buses either side of B6 (Scotland / England) joined by a line and a DC link."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-06-01", periods=3, freq="h"))
    n.add("Bus", "north", x=330000.0, y=620000.0, v_nom=400.0)   # Borders
    n.add("Bus", "south", x=380000.0, y=520000.0, v_nom=400.0)   # County Durham
    n.add("Bus", "south2", x=381000.0, y=519000.0, v_nom=400.0)
    n.add("Line", "cross", bus0="north", bus1="south", x=1.0, s_nom=2000.0)
    n.add("Line", "inside", bus0="south", bus1="south2", x=1.0, s_nom=2000.0)
    n.add("Link", "hvdc", bus0="south2", bus1="north", p_nom=2000.0, carrier="DC", p_min_pu=-1)
    return n


def test_crossings_are_signed_by_side():
    geom = load_boundary_geometries(DEFAULT_GEOMETRY)["B6"]
    cr = find_boundary_crossings(_toy_network(), geom).set_index("name")
    assert set(cr.index) == {"cross", "hvdc"}
    # bus0 of the line is north, bus0 of the link is south: opposite signs
    assert cr.at["cross", "sign"] == -cr.at["hvdc", "sign"]


def test_specs_apply_capability_times_season():
    n = _toy_network()
    cfg = {"enabled": True, "source": "secured_capability",
           "secured_capability": {"capability_year": 2030, "boundaries": ["B6"]}}
    (spec,) = build_secured_boundary_specs(n, cfg, {"modelled_year": 2030})
    assert spec["boundary_name"] == "ETYS_B6"
    assert spec["limit_series"].iloc[0] == pytest.approx(15354 * 0.70)   # June: summer outage, thermal
    lines = spec["positive_lines"] + spec["negative_lines"]
    assert lines == ["cross"] and [l["name"] for l in spec["links"]] == ["hvdc"]


def test_capability_year_override_and_exclude():
    n = _toy_network()
    cfg = {"enabled": True, "source": "secured_capability",
           "secured_capability": {"capability_year": 2030, "boundaries": ["B6"],
                                  "capability_year_overrides": {"B6": 2031},
                                  "exclude": {"B6": ["hvdc"]}}}
    (spec,) = build_secured_boundary_specs(n, cfg, {})
    assert spec["links"] == []
    assert spec["limit_series"].iloc[0] == pytest.approx(15354 * 0.70)   # 2031 = 15354 too


# Historic outage derate and capability phase-in (9 Oct 2026).

def _table(rows, boundary="B6", ratio=1.0):
    """A derate table for June 2030 hours 0-2: rows = factors for those three hours."""
    return pd.DataFrame({"boundary": boundary, "group": "SCOTEX", "month": 6, "day": 1, "hour": [0, 1, 2],
                         "factor": rows, "na1_ratio": ratio})


def _spec(n, tmp_path, table, **derate):
    path = tmp_path / "derate.csv"
    table.to_csv(path, index=False)
    cfg = {"enabled": True, "source": "secured_capability",
           "secured_capability": {"capability_year": 2030, "boundaries": ["B6"],
                                  "historic_derate": {"enabled": True, "file": str(path), **derate}}}
    (spec,) = build_secured_boundary_specs(n, cfg, {})
    return spec["limit_series"]


def test_replay_replaces_the_noa_factor(tmp_path):
    limit = _spec(_toy_network(), tmp_path, _table([0.5, 0.6, 1.0]), mode="replay")
    assert list(limit) == pytest.approx([15354 * 0.5, 15354 * 0.6, 15354 * 1.0])


def test_forward_scales_only_the_derate_beyond_noa(tmp_path):
    # June is NOA summer outage (0.70): 0.5 is 0.2 deeper, half of it kept; 0.8 is above NOA and stays.
    limit = _spec(_toy_network(), tmp_path, _table([0.5, 0.8, 0.7], ratio=0.5), mode="forward")
    assert list(limit) == pytest.approx([15354 * 0.6, 15354 * 0.8, 15354 * 0.7])


def test_unmapped_boundary_takes_its_limit_type_default_only_when_asked(tmp_path):
    table = _table([0.6, 0.6, 0.6], boundary="_default_thermal")
    assert list(_spec(_toy_network(), tmp_path, table, mode="replay", unmapped="default")) == pytest.approx([15354 * 0.6] * 3)
    assert list(_spec(_toy_network(), tmp_path, table, mode="replay")) == pytest.approx([15354 * 0.70] * 3)


def test_a_table_missing_snapshot_hours_is_refused(tmp_path):
    table = _table([0.5, 0.6, 1.0]).iloc[:2]
    with pytest.raises(ValueError, match="misses 1 snapshot"):
        _spec(_toy_network(), tmp_path, table, mode="replay")


def test_phase_in_holds_the_earlier_capability_until_the_date():
    n = _toy_network()
    cap29 = load_capabilities(DEFAULT_CAPABILITY, 2029, "HT").loc["B6", "capability_mw"]
    cfg = {"enabled": True, "source": "secured_capability",
           "secured_capability": {"capability_year": 2030, "boundaries": ["B6"],
                                  "phase_in": {"enabled": True, "before_year": 2029, "from": "2030-06-01 01:00"}}}
    (spec,) = build_secured_boundary_specs(n, cfg, {})
    assert list(spec["limit_series"]) == pytest.approx([cap29 * 0.70, 15354 * 0.70, 15354 * 0.70])
    # A boundary with a year override keeps it: phase-in does not apply.
    cfg["secured_capability"]["capability_year_overrides"] = {"B6": 2030}
    (spec,) = build_secured_boundary_specs(n, cfg, {})
    assert list(spec["limit_series"]) == pytest.approx([15354 * 0.70] * 3)
