"""Study folders: list `inherits` (decision D8), month families
expanded by the loader, and family names in selections."""

from pathlib import Path

import pytest
import yaml

from config.config_loader import (MONTH_NAMES, expand_families, expand_selection, load_scenario_entries,
                                  read_scenario_sources)


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


@pytest.fixture
def config_dir(tmp_path):
    d = tmp_path / "config"
    _write(d / "config.yaml", {"studies": ["demo"], "run_scenarios": ["calib"]})
    _write(d / "scenarios.yaml", {"Single": {"modelled_year": 2020}})
    study = d / "studies" / "demo"
    _write(study / "study.yaml", {"label": "Demo 2030", "year": 2030, "family_id": "DEMO_{Mon}{year}_{family}"})
    _write(study / "templates.yaml", {
        "base_market": {"modelled_year": 2030, "market": {"enabled": True, "wholesale": {"mode": "rolling"}}},
        "bm_calibrated": {"market": {"balancing": {"window_hours": 24}}},
        "network_late": {"network": {"case": "late"}},
        "other_mode": {"market": {"wholesale": {"mode": "single"}}},
    })
    _write(study / "families.yaml", {
        "s1": {"inherits": ["base_market"]},
        "calib": {"inherits": ["base_market", "bm_calibrated"], "stage1": "s1",
                  "month_overrides": {"Mar": {"market": {"balancing": {"window_hours": 12}}}}},
        "late": {"inherits": ["base_market", "bm_calibrated", "network_late"], "months": ["Jan", "Feb"],
                 "network_from": "DEMO_build"},
    })
    _write(study / "scenarios.yaml", {"DEMO_build": {"inherits": "base_market"}})
    return d


@pytest.mark.unit
def test_a_family_expands_to_its_months_with_periods_and_stage1(config_dir):
    entries = load_scenario_entries(config_dir)
    months = [f"DEMO_{m}2030_calib" for m in MONTH_NAMES]
    assert all(sid in entries for sid in months)
    feb = entries["DEMO_Feb2030_calib"]
    assert feb["solve_period"] == {"start": "2030-02-01 00:00", "end": "2030-02-28 23:00"}
    assert feb["wholesale_from"] == "DEMO_Feb2030_s1"
    assert feb["market"]["balancing"]["window_hours"] == 24
    assert entries["DEMO_Mar2030_calib"]["market"]["balancing"]["window_hours"] == 12      # month override
    assert {"DEMO_Jan2030_late", "DEMO_Feb2030_late"} <= set(entries) and "DEMO_Mar2030_late" not in entries
    assert entries["DEMO_Jan2030_late"]["network_from"] == "DEMO_build"
    assert entries["DEMO_Jan2030_late"]["network"]["case"] == "late"


@pytest.mark.unit
def test_list_inherits_merges_left_to_right_and_refuses_a_conflict(config_dir):
    study = config_dir / "studies" / "demo"
    families = yaml.safe_load((study / "families.yaml").read_text(encoding="utf-8"))
    families["clash"] = {"inherits": ["base_market", "other_mode"], "months": ["Jan"]}
    _write(study / "families.yaml", families)
    with pytest.raises(ValueError, match="does not depend on the order of inherits"):
        load_scenario_entries(config_dir)
    families["clash"]["market"] = {"wholesale": {"mode": "single"}}      # the family settles it itself
    _write(study / "families.yaml", families)
    assert load_scenario_entries(config_dir)["DEMO_Jan2030_clash"]["market"]["wholesale"]["mode"] == "single"


@pytest.mark.unit
def test_a_family_name_in_a_selection_expands_to_exactly_its_months(config_dir):
    sources = read_scenario_sources(config_dir)
    assert sources.families["demo:late"] == ["DEMO_Jan2030_late", "DEMO_Feb2030_late"]
    assert expand_selection(["late", "Single", "demo:s1"], sources) == (
        ["DEMO_Jan2030_late", "DEMO_Feb2030_late", "Single"] + [f"DEMO_{m}2030_s1" for m in MONTH_NAMES])


@pytest.mark.unit
def test_bad_families_are_refused():
    with pytest.raises(ValueError, match="not a family"):
        expand_families("demo", {"year": 2030, "family_id": "X_{Mon}_{family}"}, {"a": {"stage1": "missing"}})
    with pytest.raises(ValueError, match="months it does not have"):
        expand_families("demo", {"year": 2030, "family_id": "X_{Mon}_{family}"},
                        {"a": {"months": ["Jan"], "month_overrides": {"Feb": {}}}})
    with pytest.raises(ValueError, match="no id"):
        expand_families("demo", {"year": 2030}, {"a": {}})


def test_studies_env_replaces_the_listed_studies(tmp_path, monkeypatch):
    """PYPSA_GB_STUDIES: ``all`` adds every live study except drafts; a comma list replaces config.yaml's list."""
    from config.config_loader import STUDIES_ENV, main_config_with_archive, read_main_config

    d = tmp_path / "config"
    _write(d / "config.yaml", {"studies": [], "run_scenarios": []})
    _write(d / "studies" / "alpha" / "study.yaml", {"label": "A", "year": 2030})
    _write(d / "studies" / "beta" / "study.yaml", {"label": "B", "year": 2045, "draft": True})
    (d / "studies" / "archive" / "old").mkdir(parents=True)

    monkeypatch.delenv(STUDIES_ENV, raising=False)
    assert read_main_config(d)["studies"] == []
    monkeypatch.setenv(STUDIES_ENV, "all")
    assert read_main_config(d)["studies"] == ["alpha"]
    assert main_config_with_archive(d)["studies"] == ["alpha", "archive/old"]
    monkeypatch.setenv(STUDIES_ENV, "beta, alpha")
    assert read_main_config(d)["studies"] == ["beta", "alpha"]
