"""The defaults' layers (config/defaults/README.md): every axis value has its file, the method overlay holds no
source data, the two overlays never set the same key, and --explain attributes every value."""

import pytest

from config.config_loader import (BUILD_METHODS, DISPATCH_MODES, SUPPORTED_TARGET_METHODS, TARGET_SOURCES,
                                  defaults_dir, defaults_files, explain_scenario, load_layer)

#: Keys that name a publication's edition or pathway: source data, never method.
SOURCE_KEYS = {"FES_year", "fes_year", "FES_scenario", "fes_scenario", "CP30_pathway", "ed1_component_pathway",
               "peak_target"}
#: Strings that point at one publication's files or tables.
SOURCE_MARKERS = ("cp30", "fes24", "fes25", "es1", "ed1")


def _leaves(node, path=()):
    if isinstance(node, dict) and node:
        for key, value in node.items():
            yield from _leaves(value, path + (key,))
    else:
        yield path, node


@pytest.mark.unit
def test_every_axis_value_has_its_overlay_file():
    root = defaults_dir()
    for method in BUILD_METHODS:
        assert (root / "method" / f"{method}.yaml").exists(), method
    for source in TARGET_SOURCES:
        assert (root / "source" / f"{source}.yaml").exists(), source
    for mode in DISPATCH_MODES:                      # self is the market itself: no overlay
        assert (root / "dispatch" / f"{mode}.yaml").exists() == (mode != "self"), mode
    files = {p.resolve() for p in defaults_files()}
    assert all(p.resolve() in files for p in root.glob("*/*.yaml"))


@pytest.mark.unit
@pytest.mark.parametrize("method", BUILD_METHODS)
def test_the_method_overlay_holds_no_source_data(method):
    for path, value in _leaves(load_layer("method", method)):
        assert not SOURCE_KEYS & set(path), ".".join(path)
        if isinstance(value, str):
            assert not any(m in value.lower() for m in SOURCE_MARKERS), (".".join(path), value)


@pytest.mark.unit
@pytest.mark.parametrize("source,method", sorted(SUPPORTED_TARGET_METHODS))
def test_the_two_overlays_never_set_the_same_key(source, method):
    method_paths = {p for p, _ in _leaves(load_layer("method", method)) if p}
    source_paths = {p for p, _ in _leaves(load_layer("source", source)) if p}
    clash = sorted(".".join(p) for p in method_paths
                   for q in source_paths if p[:len(q)] == q or q[:len(p)] == p)
    assert not clash, clash[:5]


@pytest.mark.unit
def test_the_dispatch_overlay_switches_only_market_solve_and_support_settings():
    # a mode switch applied after method and source: it may override their keys, but stays in its own lane
    for mode in DISPATCH_MODES:
        if mode == "self":
            continue
        tops = {path[0] for path, _ in _leaves(load_layer("dispatch", mode)) if path}
        assert tops <= {"market", "solve", "support"}, tops


@pytest.mark.unit
@pytest.mark.parametrize("scenario_id", [
    "CP30_FFR_2030_Jan_cp30_network",
    "FES25_HT_2050_Jan_market",
    "Validation_2020",
    "CP30_FFR_2030_Jan_cp30_network_central_dispatch",
])
def test_explain_attributes_every_value_to_a_layer(scenario_id):
    out = explain_scenario(scenario_id)
    labels = set(out["layers"]) | {"loader"}
    assert all(label in labels for _, _, label in out["values"])
    derived = {path for path, _, label in out["values"] if label == "loader"}
    # what the loader derives: targets and method for a legacy id, or the old keys a targets-only entry gets filled
    assert derived <= {"targets.source", "targets.edition", "targets.pathway", "method", "_scenario_id",
                       "scenario_source", "CP30_pathway", "FES_year", "FES_scenario"}, derived
