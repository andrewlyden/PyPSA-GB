"""Stage 4 of the year-generic workflow: ``targets`` and ``method`` replace ``is_cp30_scenario``.

The truth table: for every scenario the loader knows, every gate the old predicates decided
gives the same answer under ``target_source()`` / ``uses_brownfield()``. The old predicates are
copied here frozen, as they stood before stage 4 (26 Oct 2026), so the comparison does not move
when the code does. Loading every scenario takes about a minute.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest

from config.config_loader import (
    load_config,
    load_scenario_entries,
    resolve_targets_and_method,
)
from scripts.utilities.scenario_detection import (
    build_method,
    is_historical_scenario,
    target_source,
    uses_brownfield,
    uses_gsp_blocks,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── Frozen references (the code before stage 4) ─────────────────────────────


def _frozen_is_cp30_scenario(scenario: dict) -> bool:
    """scripts/utilities/scenario_detection.is_cp30_scenario: every rule gate in generators,
    hydrogen, interconnectors and storage .smk."""
    if is_historical_scenario(scenario):
        return False
    source = scenario.get("scenario_source") or scenario.get("source") or scenario.get("data_source") or ""
    return str(source).upper() == "CP30" or "CP30_pathway" in scenario


def _frozen_cp30_defaults_attach(entry: dict) -> bool:
    """config_loader._is_cp30_scenario on the scenario's own entry: CP30_defaults attach."""
    source = entry.get("scenario_source") or entry.get("source") or entry.get("data_source") or ""
    return str(source).upper() == "CP30" or "CP30_pathway" in entry


def _frozen_scenario_source_equals(scenario: dict) -> bool:
    """``scenario_config.get('scenario_source') == 'CP30'``: integrate_thermal_generators,
    integrate_renewable_generators (future path), add_storage, CP30/assumptions.is_cp30_config."""
    return scenario.get("scenario_source") == "CP30"


def _frozen_scenario_source_upper(scenario: dict) -> bool:
    """``str(scenario.get('scenario_source', '')).upper() == 'CP30'``: the large hydro inflow,
    rules/CP30.smk Stage 1 families, the Snakefile CP30 validation targets."""
    return str(scenario.get("scenario_source", "")).upper() == "CP30"


# rules/demand.smk scenario_is_cp30 had no historical test: the same as the attach rule.
_frozen_demand_scenario_is_cp30 = _frozen_cp30_defaults_attach


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def all_loaded():
    # every scenario id, archived families included (decision D9)
    config = load_config(include_archive=True)
    return config["scenarios"], load_scenario_entries(include_archive=True)


@pytest.fixture(scope="module")
def loaded(all_loaded):
    """The frozen legacy ids, which still derive targets and method from the old keys: the truth table and the
    derivation table are about them. Newer ids write both (test_new_ids_write_targets_and_method)."""
    from config.config_loader import legacy_scenario_ids
    legacy = legacy_scenario_ids()
    scenarios, entries = all_loaded
    return ({sid: sc for sid, sc in scenarios.items() if sid in legacy},
            {sid: e for sid, e in entries.items() if sid in legacy})


def test_new_ids_write_targets_and_method(all_loaded):
    from config.config_loader import legacy_scenario_ids
    legacy = legacy_scenario_ids()
    scenarios, entries = all_loaded
    new = [sid for sid in scenarios if sid not in legacy]
    assert new, "the release ids (CP30_FFR_2030_<Mon>_<family>, ...) are not legacy"
    for sid in new:
        assert "targets" in entries[sid] and "method" in entries[sid], sid


# ── The truth table ──────────────────────────────────────────────────────────


def test_every_gate_unchanged_for_every_scenario(loaded):
    scenarios, entries = loaded
    assert len(scenarios) > 1000
    mismatches = []
    for sid, merged in scenarios.items():
        entry = entries[sid]
        old = _frozen_is_cp30_scenario(merged)
        old_variants = {
            "cp30_defaults_attach": _frozen_cp30_defaults_attach(entry),
            "demand_scenario_is_cp30": _frozen_demand_scenario_is_cp30(merged),
            "scenario_source_equals": _frozen_scenario_source_equals(merged),
            "scenario_source_upper": _frozen_scenario_source_upper(merged),
        }
        # Without a historical CP30 scenario the old predicates all agree; check it, since the
        # replacement below relies on it.
        for name, value in old_variants.items():
            if value != old:
                mismatches.append((sid, f"old {name} != is_cp30_scenario"))
        # The same answers on the merged config (written or derived keys) and on a dict
        # without them (the fallback for dicts built outside config_loader).
        stripped = {k: v for k, v in merged.items() if k not in ("targets", "method")}
        for label, scenario in (("merged", merged), ("stripped", stripped)):
            source_gate = target_source(scenario) == "CP30"
            method_gate = uses_brownfield(scenario)
            if source_gate != old:
                mismatches.append((sid, f"{label}: target_source gate {source_gate} != {old}"))
            if method_gate != old:
                mismatches.append((sid, f"{label}: uses_brownfield gate {method_gate} != {old}"))
            historical = is_historical_scenario(scenario)
            if (target_source(scenario) == "historical") != historical:
                mismatches.append((sid, f"{label}: historical source != is_historical_scenario"))
            if uses_gsp_blocks(scenario) != (not historical and not old):
                mismatches.append((sid, f"{label}: uses_gsp_blocks wrong"))
        # The brownfield defaults attached exactly where CP30_defaults attached before.
        if (merged["method"] == "brownfield") != old_variants["cp30_defaults_attach"]:
            mismatches.append((sid, "CP30_defaults attach changed"))
    assert not mismatches, mismatches[:20]


def test_derivation_table(loaded):
    """The families today's keys derive to (documented in config_loader and AGENTS.md section 6)."""
    from config.config_loader import _legacy_targets, legacy_target_source
    scenarios, entries = loaded
    # The canonical scenarios (config/scenarios.yaml) write both keys as examples to copy; what they write must be
    # what their year or old keys derive. Every other legacy id still derives both.
    canonical = set(load_scenario_entries(PROJECT_ROOT / "config", main_config={"studies": []}))
    table = Counter()
    for sid, merged in scenarios.items():
        if sid in canonical and "targets" in entries[sid]:
            assert merged["targets"] == _legacy_targets(legacy_target_source(merged), merged, sid), sid
        else:
            assert "targets" not in entries[sid] and "method" not in entries[sid], sid
        t = merged["targets"]
        assert set(t) == {"source", "edition", "pathway"}, sid
        assert merged["method"] == build_method(merged)
        table[(t["source"], t["edition"], t["pathway"], merged["method"])] += 1
    sources = {key[0] for key in table}
    assert sources == {"CP30", "FES", "historical"}
    assert all(method == "brownfield" for (source, _, _, method) in table if source == "CP30")
    assert all(method == "gsp_block" for (source, _, _, method) in table if source == "FES")
    assert all(edition == 2024 for (source, edition, _, _) in table if source == "CP30")
    assert scenarios["HT35"]["targets"] == {"source": "FES", "edition": 2024, "pathway": "HT"}
    assert scenarios["HT35"]["method"] == "gsp_block"
    cp30 = scenarios["CP30_FFR_2030_ETYS_market_Jan2030"]
    assert cp30["targets"] == {"source": "CP30", "edition": 2024, "pathway": "FFR"}
    assert cp30["method"] == "brownfield"
    assert scenarios["Historical_2020_reduced"]["targets"] == {
        "source": "historical", "edition": None, "pathway": None}
    assert scenarios["Historical_2020_reduced"]["method"] == "historical"


def test_no_is_cp30_scenario_left():
    """Rules, the Snakefiles and the scripts gate on target_source() / uses_brownfield() only."""
    pattern = re.compile(r"\bis_cp30_scenario\b|\bscenario_is_cp30\b|\b_is_cp30_scenario\b")
    files = [*PROJECT_ROOT.glob("Snakefile*"), *(PROJECT_ROOT / "rules").glob("*.smk"),
             PROJECT_ROOT / "config" / "config_loader.py"]
    files += [p for p in (PROJECT_ROOT / "scripts").rglob("*.py") if "archive" not in p.parts]
    hits = [str(p.relative_to(PROJECT_ROOT)) for p in files
            if pattern.search(re.sub(r"(?m)#.*$", "", p.read_text(encoding="utf-8", errors="replace")))
            and p.name != "scenario_detection.py"]
    assert not hits, hits


# ── The loader's rules for written keys ──────────────────────────────────────


def _resolve(entry: dict, defaults: dict | None = None, scenario_id: str = "test", legacy_ids=("test",)):
    return resolve_targets_and_method(entry, {**(defaults or {}), **entry}, scenario_id, legacy_ids=legacy_ids)


def test_new_study_must_state_method():
    with pytest.raises(ValueError, match="must also write method"):
        _resolve({"modelled_year": 2040, "targets": {"source": "FES", "edition": 2025, "pathway": "HT"}})


def test_future_scenario_without_any_source_is_refused():
    with pytest.raises(ValueError, match="states no targets"):
        _resolve({"modelled_year": 2040})


def test_targets_only_fills_the_old_keys():
    targets, method, fill = _resolve({"modelled_year": 2040, "method": "gsp_block",
                                      "targets": {"source": "FES", "edition": 2025, "pathway": "HT"}})
    assert targets == {"source": "FES", "edition": 2025, "pathway": "HT"} and method == "gsp_block"
    assert fill == {"FES_year": 2025, "FES_scenario": "Holistic Transition"}
    targets, method, fill = _resolve({"modelled_year": 2028, "method": "brownfield",
                                      "targets": {"source": "CP30", "pathway": "FFR"}})
    assert targets == {"source": "CP30", "edition": 2024, "pathway": "FFR"} and method == "brownfield"
    assert fill == {"scenario_source": "CP30", "CP30_pathway": "Further Flex and Renewables"}


def test_written_method_wins_over_derivation():
    targets, method, fill = _resolve({"modelled_year": 2035, "FES_year": 2024,
                                      "FES_scenario": "Holistic Transition", "method": "gsp_block"})
    assert targets["source"] == "FES" and method == "gsp_block" and fill == {}


def test_targets_must_agree_with_old_keys():
    with pytest.raises(ValueError, match="disagree"):
        _resolve({"modelled_year": 2035, "FES_year": 2024, "FES_scenario": "Holistic Transition",
                  "method": "gsp_block", "targets": {"source": "FES", "edition": 2025, "pathway": "HT"}})


def test_historical_year_needs_historical_targets():
    with pytest.raises(ValueError):
        _resolve({"modelled_year": 2020, "method": "brownfield",
                  "targets": {"source": "CP30", "pathway": "FFR"}})


def test_unsupported_pair_is_refused_until_stage_7():
    with pytest.raises(ValueError, match="not supported yet"):
        _resolve({"modelled_year": 2032, "method": "brownfield",
                  "targets": {"source": "FES", "edition": 2025, "pathway": "HT"}})


# ── Stage 5 carry-forward: frozen legacy ids, registry pathways ──────────────


def test_a_new_scenario_with_only_the_old_keys_is_refused():
    entry = {"modelled_year": 2040, "FES_year": 2025, "FES_scenario": "Holistic Transition"}
    with pytest.raises(ValueError, match="new scenario must write targets"):
        _resolve(entry, scenario_id="FES25HT_2040", legacy_ids=())
    # A frozen id still derives them; a new id that writes targets and method is fine.
    assert _resolve(entry, scenario_id="FES25HT_2040", legacy_ids=("FES25HT_2040",))[0]["pathway"] == "HT"
    targets, method, _ = _resolve({"modelled_year": 2040, "method": "gsp_block",
                                   "targets": {"source": "FES", "edition": 2025, "pathway": "HT"}},
                                  scenario_id="FES25HT_2040", legacy_ids=())
    assert targets["edition"] == 2025 and method == "gsp_block"


def test_frozen_id_list_holds_todays_scenarios():
    from config.config_loader import legacy_scenario_ids
    ids = legacy_scenario_ids()
    assert len(ids) == 1631
    assert {"HT50_2050", "HT50_2050_Jan", "Test_Future_2030", "Historical_2020_reduced"} <= ids


def test_pathway_codes_come_from_the_registry():
    from config import config_loader as cl
    from scripts.sources import registry
    assert cl.CP30_PATHWAYS == {p.name for p in registry.pathways("CP30")}
    assert not hasattr(cl, "PATHWAY_CODES")
    assert cl._pathway_code("FES", "Holistic Transition", "t", 2025) == "HT"
    assert cl._pathway_code("FES", "HT", "t", 2025) == "HT"
    # A written target is checked against its own edition ...
    with pytest.raises(ValueError, match="unknown FES 2024 pathway"):
        cl._pathway_code("FES", "Leading the Way", "t", 2024)
    # ... a frozen legacy id may name another edition's pathway (Test_Future_2030: FES 2024, LW).
    assert cl._pathway_code("FES", "Leading the Way", "t", 2024, any_edition=True) == "LW"


def test_every_scenario_resolves_its_pathway_through_the_registry():
    from scripts.sources import registry
    scenarios = load_config()["scenarios"]
    for sid, cfg in scenarios.items():
        targets = cfg["targets"]
        if targets["source"] == "historical" or targets["pathway"] is None:
            continue
        codes = {p.code for e in registry.editions(targets["source"]) for p in registry.pathways(targets["source"], e)}
        assert targets["pathway"] in codes, sid
